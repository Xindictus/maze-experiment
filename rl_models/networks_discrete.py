import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# why retain graph? Do not auto free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer:
    """
    Convert to numpy
    """

    def __init__(self, args):
        self.args = args
        self.storage = deque(maxlen=self.args.buffer_size)
        self.memory_size = self.args.buffer_size

        if self.args.ere:
            self.eta = self.args.eta
            self.cmin = self.args.cmin
            self.first_update = True

        if self.args.leb:
            self.merge_buffers(self.args.buffer_path)
        if self.args.dqfd:
            self.load_demostration(self.args.demo_path)

    # add the samples
    def add(self, obs, action, reward, obs_, done, transition_info):
        data = (obs, action, reward, obs_, done, transition_info)
        self.storage.append(data)

    def get_size(self):
        return len(self.storage)

    # encode samples
    def _encode_sample(self, idx, demo=False):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            if demo:
                data = self.expert_storage[i]
            else:
                data = self.storage[i]
            obs, action, reward, obs_, done, transition_info = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return (
            np.array(obses),
            np.array(actions),
            np.array(rewards),
            np.array(obses_),
            np.array(dones),
            np.array(transition_info),
        )

    # sample from the memory
    def sample(self, block_nb, batch_size, total_updates=1000):
        if self.args.dqfd:
            splits = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0]
            # print('Executing DQfD with split: ',splits[block_nb])
            dem_indexes = [
                random.randint(0, len(self.expert_storage) - 1)
                for _ in range(int(batch_size * splits[block_nb]))
            ]
            dem_data = self._encode_sample(dem_indexes, demo=True)

            indexes = [
                random.randint(0, len(self.storage) - 1)
                for _ in range(int(batch_size * (1 - splits[block_nb])))
            ]
            if len(indexes) != 0:
                data = self._encode_sample(indexes)

                obses = np.concatenate((dem_data[0], data[0]), axis=0)
                actions = np.concatenate((dem_data[1], data[1]), axis=0)
                rewards = np.concatenate((dem_data[2], data[2]), axis=0)
                obses_ = np.concatenate((dem_data[3], data[3]), axis=0)
                dones = np.concatenate((dem_data[4], data[4]), axis=0)
                transition_info = np.concatenate(
                    (dem_data[5], data[5]), axis=0
                )

            elif len(indexes) == 0:
                obses = dem_data[0]
                actions = dem_data[1]
                rewards = dem_data[2]
                obses_ = dem_data[3]
                dones = dem_data[4]
                transition_info = dem_data[5]
            # print(transition_info)
            return obses, actions, rewards, obses_, dones, transition_info
        else:
            if self.args.ere:
                n = len(self.storage)
                if self.first_update:
                    c_k = n
                    self.first_update = False
                else:
                    update_num = len(self.storage)
                    c_k = max(
                        int(
                            n
                            * (self.eta ** (update_num * 1000 / total_updates))
                        ),
                        self.cmin,
                    )
                idxes_range = range(max(n - c_k, 0), n)
                idxes = random.sample(
                    idxes_range, min(batch_size, len(idxes_range))
                )
                return self._encode_sample(idxes)
            else:
                idxes = [
                    random.randint(0, len(self.storage) - 1)
                    for _ in range(batch_size)
                ]
                return self._encode_sample(idxes)

    # Save Buffer
    def save_buffer(self, path, name):
        path = os.path.join(path, name)
        self.storage = np.array(self.storage, dtype=object)
        np.save(path, self.storage)

    # Load Buffer
    def load_buffer(self, path):
        self.storage = np.load(path, allow_pickle=True).tolist()

    def merge_buffers(self, path):
        files = os.listdir(path)
        buffers = []
        for file in files:
            buffers.append(
                np.load(os.path.join(path, file), allow_pickle=True).tolist()
            )
        temp_storage = []
        for buffer in buffers:
            temp_storage += buffer
        print("Merged Buffers size:", len(temp_storage))
        self.storage = deque(maxlen=len(temp_storage))
        for data in temp_storage:
            self.storage.append(data)

    def load_demostration(self, path):
        self.expert_storage = np.load(path, allow_pickle=True).tolist()
        print("Demonstration Buffer size:", len(self.expert_storage))


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_units,
        name="actor",
        chkpt_dir="tmp/sac",
        load_file=None,
    ):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac'+ str(date.today()))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.name = name
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[1], action_dim),
        ).apply(init_weights)

    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()

        arg_max_action = torch.argmax(actions_probs)

        return action.item(), arg_max_action.item()

    def save_checkpoint(self, block):
        torch.save(
            self.state_dict(),
            os.path.join(self.checkpoint_dir, str(block) + "_actor.pt"),
        )

    def load_checkpoint(self):
        self.load_state_dict(
            torch.load(
                os.path.join(self.load_file, "_actor.pt"), map_location=device
            )
        )


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_units,
        name="critic",
        chkpt_dir="tmp/sac",
        load_file=None,
    ):
        super(Critic, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac'+ str(date.today()))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.load_file = load_file

        # self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units)
        # self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units)
        self.qnet1 = QNet(state_dim, action_dim, n_hidden_units)
        self.qnet2 = QNet(state_dim, action_dim, n_hidden_units)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2

    def sample_qvalue(self, s):
        s = torch.from_numpy(s).float().to(device)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        print("Q1:", q1)
        q1 = q1.detach().cpu().numpy()
        q2 = q2.detach().cpu().numpy()
        Q = []
        for i in range(len(q1)):
            Q.append(min(q1[i], q2[i]))
        return Q

    def save_checkpoint(self, block):
        torch.save(
            self.state_dict(),
            os.path.join(self.checkpoint_dir, str(block) + "_critic.pt"),
        )

    def load_checkpoint(self):
        self.load_state_dict(
            torch.load(
                os.path.join(self.load_file, "_critic.pt"), map_location=device
            )
        )


class QNet(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_units,
        name="QNet",
        chkpt_dir="tmp/sac",
    ):
        super(QNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[1], action_dim),
        ).apply(init_weights)

    def forward(self, s):
        q_value = self.shared_mlp(s)

        return q_value


class DuelQNet(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_units,
        name="DuelQNet",
        chkpt_dir="tmp/sac",
    ):
        super(DuelQNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
        ).apply(init_weights)

        # self.q_head = nn.Linear(n_hidden_units, action_dim)

        self.action_head = nn.Linear(n_hidden_units[1], action_dim).apply(
            init_weights
        )
        self.value_head = nn.Linear(n_hidden_units[1], 1).apply(init_weights)

    def forward(self, s):
        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
        # return self.q_head(s)
