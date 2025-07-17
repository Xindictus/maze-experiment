from typing import Dict

import torch as T
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import Trainer
from src.marl.algos.qmix import MAC
from src.marl.buffers.replay_buffer_base import ReplayBufferBase
from src.marl.mixing.qmix import QMixer


class QmixTrainer(Trainer):
    def __init__(
        self,
        buffer: ReplayBufferBase,
        mac: MAC,
        mixer: QMixer,
        target_mac: MAC,
        target_mixer: QMixer,
        config: QmixBaseConfig,
    ):
        self.buffer = buffer
        self.mac = mac
        self.mixer = mixer
        self.target_mac = target_mac
        self.target_mixer = target_mixer
        self.config = config

        self.mixer.to(config.device)
        self.target_mixer.to(config.device)

        # Optimizer on all trainable params
        self.params = list(self.mac.parameters()) + list(
            self.mixer.parameters()
        )
        self.optimizer = optim.RMSprop(
            params=self.params,
            lr=config.learning_rate,
            alpha=config.optim_alpha,
            eps=config.optim_eps,
        )

        self.training_steps = 0

    def train(self) -> None:
        batch = self.buffer.sample(self.config.batch_size)
        batch = self._to_device(batch)

        # (batch, T)
        rewards = batch["rewards"][:, :-1]

        # (batch, T, n_agents, 1)
        actions = batch["actions"][:, :-1]

        # (batch, T)
        dones = batch["dones"][:, :-1].float()

        # (batch, T)
        # TODO: Adjust buffer
        mask = batch["mask"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - dones[:, :-1])

        # (batch, T + 1, n_agents, n_actions)
        avail_actions = batch["avail_actions"]

        # Compute Q-values from mac and target_mac

        # (batch, T + 1, n_agents, n_actions)
        mac_out = self._get_q_values(self.mac, batch)

        # (batch, T, n_agents, n_actions)
        target_mac_out = self._get_q_values(self.target_mac, batch)

        # Chosen Q-values (using actions taken)
        # (batch, T, n_agents)
        chosen_qs = T.gather(mac_out[:, :-1], dim=-1, index=actions).squeeze(
            -1
        )

        # Mask out invalid actions in target Qs
        masked_target_mac_out = target_mac_out[:, 1:]
        masked_target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_max_qvals = masked_target_mac_out.max(dim=-1)[0]

        print("mac_out", mac_out[:, :-1].shape)
        print("actions", actions.shape)
        print("chosen_qs", chosen_qs.shape)
        print("state", batch["state"][:, :-1].shape)

        # Mix agent individual Qs into global Q-tot
        # (batch, T, 1)
        chosen_q_tot = self.mixer(
            chosen_qs.to(self.config.device),
            batch["state"][:, :-1].to(self.config.device),
        )

        print("target_max_qvals", target_max_qvals.shape)
        print("target_states", batch["state"][:, 1:].shape)

        target_q_tot = self.target_mixer(
            target_max_qvals.to(self.config.device),
            batch["state"][:, 1:].to(self.config.device),
        )

        print("rewards", rewards.shape)
        print("dones", dones.shape)
        print("target_q_tot", target_q_tot.shape)

        # TD target
        # (batch, T, 1)
        targets = (
            rewards + self.config.gamma * (1 - dones) * target_q_tot[:, :-1]
        )

        # TD loss
        # (batch, T, 1)
        td_error = chosen_q_tot[:, :-1] - targets.detach()
        masked_td_error = td_error * mask.unsqueeze(-1)
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.params, self.config.grad_norm_clip)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self._update_targets()

    def _get_q_values(self, mac: MAC, batch: Dict[str, T.Tensor]) -> T.Tensor:
        """
        Runs all agents through the sequence of observations.
        Returns Q-values: (batch, T + 1, n_agents, n_actions)
        """
        B, T_1, N, _ = batch["obs"].shape

        all_qs = []

        for t in range(T_1):
            q_at_t = []

            for agent_id in range(N):
                # (batch, obs_dim)
                obs = batch["obs"][:, t, agent_id, :]

                # (batch, n_actions)
                q = mac.forward(agent_id, obs)
                q_at_t.append(q)

            # (batch, n_agents, n_actions)
            q_at_t = T.stack(q_at_t, dim=1)

            all_qs.append(q_at_t)

        return T.stack(all_qs, dim=1)

    def _to_device(self, batch: dict[str, T.Tensor]) -> dict[str, T.Tensor]:
        return {k: v.to(self.config.device) for k, v in batch.items()}

    def _update_targets(self) -> None:
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
