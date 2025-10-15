from abc import ABC, abstractmethod
from typing import Optional

import torch as T
from pydantic import BaseModel

from .action_space import ActionSpace
from .agent_network import AgentNetwork
from .observation import Observation


class Agent(ABC):
    def __init__(
        self,
        action_space: ActionSpace,
        config: BaseModel,
        network: AgentNetwork,
        name: str,
    ):
        self.action_space = action_space
        self.config = config
        self.network = network
        self.observation: Observation = None
        self.name: str = name
        self.h_out = None

    @abstractmethod
    def forward(
        self, obs: T.Tensor, hidden: Optional[T.Tensor] = None
    ) -> T.Tensor:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, obs: T.Tensor, epsilon: float) -> int:
        raise NotImplementedError
