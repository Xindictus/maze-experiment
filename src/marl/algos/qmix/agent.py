import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import ActionSpace, Agent
from src.marl.algos.qmix.agent_networks import QmixNetwork


class QmixAgent(Agent):
    def __init__(
        self,
        _action_space: ActionSpace,
        _config: QmixBaseConfig,
        _network: QmixNetwork,
        # TODO: init + target + loss (configurable soft/hard update)
        _target_network: QmixNetwork
    ):
        super().__init__(
            action_space=_action_space, config=_config, network=_network
        )

    def forward(self, obs: T.Tensor) -> T.Tensor:
        self.network(obs)

    def select_action(self, obs: T.Tensor, epsilon: float) -> int:
        q_values = self.forward(obs)

        if T.rand(1).item() < epsilon:
            return T.randint(0, q_values.shape[-1], (1,)).item()

        return T.argmax(q_values, dim=-1).item()
