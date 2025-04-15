import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from rl_models.iql.buffer import ReplayBuffer
from rl_models.iql.networks import Actor, Critic, Value


class IQL(nn.Module):
    def __init__(self, args, p_name, ID="first", input_dims=(8,)):
        super(IQL, self).__init__()
        self.args = args
        self.p_name = p_name
        self.ID = ID

        self.load_file = os.path.join(
            self.args.load_model, self.args.agent_type
        )

        self.args.state_shape = input_dims[0]
        self.args.action_shape = args.num_actions

        self.freeze_agent = False

        self.state_size = self.args.state_shape
        self.action_size = self.args.action_shape

        self.device = self.args.device

        self.gamma = torch.FloatTensor([0.99]).to(self.args.device)
        self.hard_update_every = 10
        hidden_size = 32
        learning_rate = 3e-4
        self.clip_grad_param = 100
        self.temperature = torch.FloatTensor([100]).to(self.args.device)
        self.expectile = torch.FloatTensor([0.8]).to(self.args.device)

        # Actor Network
        self.actor = Actor(
            self.state_size,
            self.action_size,
            hidden_size,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_lr
        )

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(
            self.state_size,
            self.action_size,
            hidden_size,
            2,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)
        self.critic2 = Critic(
            self.state_size,
            self.action_size,
            hidden_size,
            1,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(
            self.state_size,
            self.action_size,
            hidden_size,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(
            self.state_size,
            self.action_size,
            hidden_size,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=args.critic_lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=args.critic_lr
        )

        self.value_net = Value(
            state_size=self.state_size,
            hidden_size=hidden_size,
            chkpt_dir=self.args.chkpt_dir,
            load_file=self.load_file,
        ).to(self.args.device)

        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )
        self.step = 0

        self.memory = ReplayBuffer(
            args=args,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=self.device,
        )

        self.alpha_history = []

        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

        self.entropy_history = []
        self.entropy_loss_history = []

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []

    # def get_action(self, state, eval=False):
    #     """Returns actions for given state as per current policy."""
    #     state = torch.from_numpy(state).float().to(self.device)
    #     with torch.no_grad():
    #             action = self.actor.get_action(state)
    #     return action.numpy()

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1, q2)

            self.q1_history.append(q1.mean().item())
            self.q2_history.append(q2.mean().item())

        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(
            exp_a, torch.FloatTensor([100.0]).to(states.device)
        ).squeeze(-1)

        _, dist = self.actor.evaluate(states)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        self.policy_history.append(log_probs.mean().item())
        self.policy_loss_history.append(actor_loss.mean().item())
        return actor_loss

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1, q2)

        value = self.value_net(states)
        value_loss = loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)

        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()

        self.q1_loss_history.append(critic1_loss.mean().item())
        self.q2_loss_history.append(critic2_loss.mean().item())
        return critic1_loss, critic2_loss

    def learn(self, nb_block):
        self.step += 1
        states, actions, rewards, next_states, dones = self.memory.sample(
            nb_block
        )

        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic1_loss, critic2_loss = self.calc_q_loss(
            states, actions, rewards, dones, next_states
        )

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        if self.step % self.args.update_target_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)

        return (
            actor_loss.item(),
            critic1_loss.item(),
            critic2_loss.item(),
            value_loss.item(),
        )

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def can_learn(self, block_nb):
        if self.args.dqfd:
            splits = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0]
            blk_req = int(self.args.batch_size * (1 - splits[block_nb]))
            print(blk_req, self.memory.__len__())
            if blk_req < self.memory.__len__():
                return True
            else:
                return False

        else:
            if self.args.batch_size < self.memory.__len__():
                return True
            else:
                return False

    def save_models(self, block_number):
        if self.args.chkpt_dir is not None:
            print(".... saving models ....")
            self.actor.save_checkpoint(block_number)
            self.critic1.save_checkpoint(block_number, "critic1")
            self.critic2.save_checkpoint(block_number, "critic2")
            self.value_net.save_checkpoint(block_number)
            # self.target_critic.save_checkpoint()

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint("critic1")
        self.critic2.load_checkpoint("critic2")
        self.value_net.load_checkpoint()
        # self.target_critic.load_checkpoint()
        self.hard_update(self.critic1, self.critic1_target)
        self.hard_update(self.critic2, self.critic2_target)

    def return_settings(self):
        actor_lr = self.args.actor_lr
        critic_lr = self.args.critic_lr
        hidden_size = self.args.hidden_size
        tau = self.args.tau
        gamma = self.args.gamma
        batch_size = self.args.batch_size

        target_entropy = None
        log_alpha = None
        alpha_lr = None

        return (
            self.ID,
            actor_lr,
            critic_lr,
            alpha_lr,
            hidden_size,
            tau,
            gamma,
            batch_size,
            target_entropy,
            log_alpha,
            self.freeze_agent,
        )

    def collect_data(self):
        return_data = {}

        # Collect data for plotting.
        return_data["q1"] = self.q1_history
        return_data["q2"] = self.q2_history

        return_data["policy"] = self.policy_history
        return_data["policy_loss"] = self.policy_loss_history

        return_data["q1_loss"] = self.q1_loss_history
        return_data["q2_loss"] = self.q2_loss_history

        return_data["entropy_loss"] = self.entropy_loss_history
        return_data["entropies"] = self.entropy_history
        return_data["alpha"] = self.alpha_history

        return_data["states"] = self.states_history
        return_data["actions"] = self.actions_history
        return_data["rewards"] = self.rewards_history
        return_data["next_states"] = self.next_states_history
        return_data["dones"] = self.dones_history

        # reseting arrays
        self.alpha_history = []

        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

        self.entropy_history = []
        self.entropy_loss_history = []

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []

        return return_data


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
