import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import ActionSpace, Agent
from src.marl.algos.qmix.agent_networks import QmixNetwork


class QmixAgent(Agent):
    target_network: QmixNetwork = None

    def __init__(
        self,
        action_space: ActionSpace,
        config: QmixBaseConfig,
        network: QmixNetwork,
    ):
        super().__init__(
            action_space=action_space, config=config, network=network
        )
        # Initialize target network with network as base
        self.target_network = self._build_target_network()

    def _build_target_network(self) -> QmixNetwork:
        target_net = QmixNetwork(config=self.config)
        target_net.load_state_dic(self.network.state_dict())
        target_net.to(self.config.device)
        return target_net

    def forward(self, obs: T.Tensor) -> T.Tensor:
        self.network(obs)

    def target_forward(self, obs: T.Tensor) -> T.Tensor:
        self.target_network(obs)

    def select_action(self, obs: T.Tensor, epsilon: float) -> int:
        q_values = self.forward(obs)

        if T.rand(1).item() < epsilon:
            return T.randint(0, q_values.shape[-1], (1,)).item()

        return T.argmax(q_values, dim=-1).item()

    def update_target_network(self) -> None:
        # TODO: init + target + loss (configurable soft/hard update)
        pass
