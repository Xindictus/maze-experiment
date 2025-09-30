from collections.abc import Iterator
from typing import Literal

import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import ActionSpace, Agent
from src.marl.algos.qmix.agent_networks import QmixNetwork
from src.utils.logger import Logger


class QmixAgent(Agent):
    def __init__(
        self,
        action_space: ActionSpace,
        config: QmixBaseConfig,
        network: QmixNetwork,
        name: str,
    ):
        super().__init__(
            action_space=action_space,
            config=config,
            network=network,
            name=name,
        )

    def forward(self, obs: T.Tensor) -> T.Tensor:
        return self.network(obs)

    def parameters(self, recurse: bool = True) -> Iterator[T.nn.Parameter]:
        return self.network.parameters(recurse=recurse)

    def select_action(
        self,
        obs: T.Tensor,
        epsilon: float,
        mode: Literal["test", "train"] = "test",
    ) -> int:
        Logger().debug(f"[{self.name}] Observation shape: {obs.shape}")
        Logger().debug(f"[{self.name}] Observation values: {obs}")

        q_values = self.forward(obs)

        if mode == "test":
            Logger().debug(f"[{self.name}] Q-Values: {q_values}")

        if mode == "train":
            if T.rand(1).item() < epsilon:
                return T.randint(0, q_values.shape[-1], (1,)).item()

        return T.argmax(q_values, dim=-1).item()

    def load_state(self, other: "QmixAgent") -> None:
        self.network.load_state_dict(other.network.state_dict())
