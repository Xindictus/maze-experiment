from abc import ABC, abstractmethod

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
    ):
        self.action_space = action_space
        self.config = config
        self.network = network
        self.observation: Observation = None

    @abstractmethod
    def forward(self, obs: T.Tensor) -> T.Tensor:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, obs: T.Tensor, epsilon: float) -> int:
        raise NotImplementedError
