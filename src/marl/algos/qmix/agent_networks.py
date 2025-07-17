from abc import ABC

import torch as T
import torch.nn as nn

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common.agent_network import AgentNetwork


class QmixNetwork(AgentNetwork, ABC):
    def __init__(self, config: QmixBaseConfig):
        super().__init__(config)

    def init_weights(self, m):
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

    def forward(self, obs: T.Tensor) -> T.Tensor:
        obs = obs.to(self.config.device)
        q_values = self.net(obs)
        return self.enforce_output_shape(q_values)


class QmixDuelingNetwork(QmixNetwork):
    pass


class QmixGRUNetwork(QmixNetwork):
    def __init__(self, config: QmixBaseConfig):
        super().__init__(config)

        # TODO: Initialize hidden states

        self.fc_in = nn.Linear(config.input_dim, config.hidden_dims[0])
        self.relu = nn.ReLU()
        # TODO
        self.gru = nn.GRU(
            input_size=config.hidden_dims[0],
            hidden_size=config.hidden_dims[0],
            batch_first=True,
        )
        self.fc_out = nn.Linear(config.hidden_dims[0], config.n_actions)
        self.apply(self.init_weights)
        self.to(config.device)

    def forward(self, obs: T.Tensor) -> T.Tensor:
        obs = obs.to(self.config.device)
        x = self.relu(self.fc_in(obs))
        x = x.unsqueeze(1)
        gru_out, h_out = self.gru(x)
        q_values = self.fc_out(gru_out.squeeze(1))

        return self.enforce_output_shape(q_values)


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
