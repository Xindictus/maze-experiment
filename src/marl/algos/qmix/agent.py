import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import ActionSpace, Agent
from src.marl.algos.qmix.agent_networks import QmixNetwork
from src.utils.logger import Logger


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
        target_net = type(self.network)(config=self.config)
        target_net.load_state_dict(self.network.state_dict())
        target_net.to(self.config.device)
        return target_net

    def forward(self, obs: T.Tensor) -> T.Tensor:
        return self.network(obs)

    def parameters(self):
        return self.network.parameters()

    def select_action(self, obs: T.Tensor, epsilon: float) -> int:
        Logger().debug(f"obs shape: {obs.shape}, values: {obs}")
        q_values = self.forward(obs)
        Logger().debug(f"q_values: {q_values}")

        if T.rand(1).item() < epsilon:
            return T.randint(0, q_values.shape[-1], (1,)).item()

        return T.argmax(q_values, dim=-1).item()

    def target_forward(self, obs: T.Tensor) -> T.Tensor:
        self.target_network(obs)

    def load_state(self, other: "QmixAgent") -> None:
        self.network.load_state_dict(other.network.state_dict())

    def update_target_network(self) -> None:
        # TODO: init + target + loss (configurable soft/hard update)
        pass
