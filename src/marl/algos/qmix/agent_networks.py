from abc import ABC
from typing import Optional

import torch as T
import torch.nn as nn

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common.agent_network import AgentNetwork


class QmixNetwork(AgentNetwork, ABC):
    def __init__(self, config: QmixBaseConfig):
        super().__init__(config)

    def init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            # TODO: switch mode and switch mode RELU
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def enforce_output_shape(self, q_values: T.Tensor) -> T.Tensor:
        assert q_values.shape[-1] == self.config.n_actions, (
            f"Expected action output dim {self.config.n_actions}"
            + f", got {q_values.shape[-1]}"
        )

        return q_values

    def forward(
        self, obs: T.Tensor, hidden: Optional[T.Tensor] = None
    ) -> T.Tensor:
        obs = obs.to(self.config.device)
        q_values = self.net(obs)
        return (self.enforce_output_shape(q_values), hidden)


class QmixDuelingNetwork(QmixNetwork):
    pass


class QmixGRUNetwork(QmixNetwork):
    def __init__(self, config: QmixBaseConfig):
        super().__init__(config)

        self.hidden_dim = config.hidden_dims[0]
        self.n_actions = config.n_actions
        self.n_layers = 1

        self.fc_in = nn.Linear(config.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            # batch_first=True,
            bias=True,
            dropout=0.0,
            bidirectional=False,
        )

        self.fc_out = nn.Linear(self.hidden_dim, self.n_actions)
        self.apply(self.init_weights)
        self.to(config.device)

    def init_hidden(self, batch_size: int) -> T.Tensor:
        return T.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim,
            device=self.config.device,
        )

    def forward(
        self, obs: T.Tensor, hidden: Optional[T.Tensor] = None
    ) -> T.Tensor:
        obs = obs.to(self.config.device)
        obs_dims = obs.dim()

        hidden = (
            self.init_hidden(batch_size=1)
            if hidden is None
            else hidden.to(self.config.device)
        )

        if obs_dims == 1:
            # [1, 1, obs]
            obs = obs.unsqueeze(0).unsqueeze(1)
        elif obs_dims == 3:
            # [B, T, dims]
            pass
        else:
            raise ValueError(
                f"Obs dimensions must be either [obs] or [B,obs], got {obs.shape}"
            )

        B = obs.size(0)
        x = self.relu(self.fc_in(obs))

        from src.utils.logger import Logger

        Logger().info(B)
        Logger().info(x.shape)
        Logger().info(hidden.shape)

        # [B, T, H] | [num_layers, B, H]
        gru_out, h_out = self.gru(x, hidden)

        Logger().info(gru_out.shape)
        Logger().info(h_out.shape)

        # [B, T, A]
        q = self.fc_out(gru_out)
        q = self.enforce_output_shape(q)

        Logger().info(q.shape)
        Logger().info(q)

        if B == 1:
            q = q.unsqueeze(0).unsqueeze(1)
        # import sys; sys.exit(1)

        return q, h_out


class QmixQNetNetwork(QmixNetwork):
    def __init__(self, config: QmixBaseConfig):
        super().__init__(config)

        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.n_actions),
        ).apply(self.init_weights)

        self.to(config.device)
