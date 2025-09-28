from typing import Any, Dict, Literal

import numpy as np
import torch as T
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import Trainer
from src.marl.algos.qmix import MAC
from src.marl.buffers.replay_buffer_base import ReplayBufferBase
from src.marl.mixing.qmix import QMixer
from src.utils.logger import Logger


class QmixTrainer(Trainer):
    def __init__(
        self,
        buffer: ReplayBufferBase,
        buffer_type: Literal["episode", "prioritized", "standard"],
        mac: MAC,
        mixer: QMixer,
        target_mac: MAC,
        target_mixer: QMixer,
        config: QmixBaseConfig,
    ):
        self.buffer = buffer
        self.buffer_type = buffer_type
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
        Logger().debug(f"Buffer size: {len(self.buffer)}")
        batch = self.buffer.sample(self.config.batch_size)

        Logger().debug(f"Batch: {batch}")
        Logger().debug(f"actions shape: {batch["actions"].shape}")
        Logger().debug(f"avail_actions shape: {batch["avail_actions"].shape}")
        # self.log_batch_shapes(batch)

        # For QMIX episode batch
        batch = self._to_device(batch)

        # (batch, T)
        rewards = batch["rewards"][:, :-1]

        # (batch, T, n_agents, 1)
        if self._is_episode_buffer():
            actions = batch["actions"][:, :-1]
        elif self._is_standard_buffer():
            actions = batch["actions"]

        # (batch, T)
        dones = batch["dones"][:, :-1].float()

        # (batch, T)
        # TODO: Adjust buffer
        mask = batch["mask"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - dones[:, :-1])

        # (batch, T + 1, n_agents, n_actions)
        avail_actions = batch["avail_actions"]

        if self._is_episode_buffer():
            states_input = batch["state"][:, 1:-1]
        elif self._is_standard_buffer():
            states_input = batch["state"]

        # Compute Q-values from mac and target_mac
        # (batch, T, n_agents, n_actions)
        mac_out = self._get_q_values_v2(self.mac, batch)

        # (batch, T, n_agents, n_actions)
        target_mac_out = self._get_q_values_v2(
            self.target_mac, batch, "target"
        )

        # Chosen Q-values (using actions taken)
        # (batch, T, n_agents)
        chosen_qs = T.gather(mac_out, dim=-1, index=actions).squeeze(-1)

        Logger().debug(f"MAC out: {mac_out}")
        Logger().debug(f"Chosen Qs: {chosen_qs}")
        Logger().debug(f"Actions: {actions}")
        Logger().debug(f"States: {states_input}")

        # Mask out invalid actions in target Qs
        masked_target_mac_out = target_mac_out.clone()

        if self._is_episode_buffer():
            masked_target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            target_max_qvals = masked_target_mac_out.max(dim=-1)[0]
        elif self._is_standard_buffer():
            masked_target_mac_out.masked_fill_(
                avail_actions.narrow(
                    1,
                    avail_actions.size(1) - masked_target_mac_out.size(1),
                    masked_target_mac_out.size(1),
                )
                == 0,
                -1e9,
            )
            target_max_qvals = masked_target_mac_out.max(dim=-1).values

        # Mix agent individual Qs into global Q-tot
        # (batch, T, 1)
        chosen_q_tot = self.mixer(
            chosen_qs.to(self.config.device),
            states_input.to(self.config.device),
        )

        Logger().debug(f"target_max_qvals (shape): {target_max_qvals.shape}")
        Logger().debug(f"target_states (shape): {batch["state"][:, 1:].shape}")

        if self._is_episode_buffer():
            target_q_tot = self.target_mixer(
                target_max_qvals[:, :-1].to(self.config.device),
                states_input.to(self.config.device),
            )
        elif self._is_standard_buffer():
            target_q_tot = self.target_mixer(
                target_max_qvals.to(self.config.device),
                states_input.to(self.config.device),
            )

        Logger().debug(f"rewards (shape): {rewards.shape}")
        Logger().debug(f"dones (shape): {dones.shape}")
        Logger().debug(f"target_q_tot: {target_q_tot.shape}")

        # TD target
        # (batch, T, 1)
        targets = rewards + self.config.gamma * (1 - dones) * target_q_tot

        # TD loss
        # (batch, T, 1)
        td_error = chosen_q_tot - targets.detach()

        # Align time dim to td_error
        if mask.size(1) != td_error.size(1):
            mask = mask[:, : td_error.size(1)]

        masked_td_error = td_error * mask
        # masked_td_error = td_error * mask.unsqueeze(-1)
        loss = (masked_td_error**2).sum() / mask.sum()

        Logger().debug(f"Loss: {loss}")

        # Optimizer
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.params, self.config.grad_norm_clip)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self._update_targets()

        return loss

    def _is_episode_buffer(self):
        return self.buffer_type == "episode"

    def _is_standard_buffer(self):
        return self.buffer_type == "standard"

    def _get_q_values_v1(
        self, mac: MAC, batch: Dict[str, T.Tensor]
    ) -> T.Tensor:
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

    # TODO: Think of stride in sliding windows
    def _get_q_values_v2(
        self,
        mac: MAC,
        batch: Dict[str, T.Tensor],
        mode: Literal["regular", "target"] = "regular",
    ) -> T.Tensor:
        """
        Runs all agents through the sequence of observations.
        Returns Q-values: (batch, T + 1, n_agents, n_actions)
        """
        B, T_1, N, _ = batch["obs"].shape
        Tlen = T_1 - 1

        if mode == "regular":
            indices = range(0, Tlen)
        elif mode == "target":
            indices = range(1, T_1)
        else:
            raise ValueError("Unknown mode for getting Q-values")

        all_qs = []

        for t in indices:
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

    def _to_device(self, batch: dict[str, Any]) -> dict[str, T.Tensor]:
        result = {}

        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                tensor = T.tensor(
                    v, dtype=T.float32 if v.dtype == np.float32 else T.long
                )
            elif isinstance(v, T.Tensor):
                tensor = v
            else:
                raise TypeError(f"Unsupported type for key '{k}': {type(v)}")

            result[k] = tensor.to(self.config.device)

        return result

    def _update_targets(self) -> None:
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _log_batch_shapes(self, batch: dict[str, T.Tensor]):
        Logger().debug("Batch tensor shapes:")

        for k, v in batch.items():
            if isinstance(v, T.Tensor):
                Logger().debug(f"  {k}: {tuple(v.shape)} | dtype: {v.dtype}")
            elif isinstance(v, np.ndarray):
                Logger().debug(f"  {k}: {v.shape} | numpy | dtype: {v.dtype}")
            else:
                Logger().debug(f"  {k}: UNKNOWN TYPE {type(v)}")
