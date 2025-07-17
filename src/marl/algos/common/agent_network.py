from abc import ABC, abstractmethod

import torch as T
import torch.nn as nn
from pydantic import BaseModel


class AgentNetwork(nn.Module, ABC):
    def __init__(self, config: BaseModel):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, obs: T.Tensor) -> T.Tensor:
        # Returns the Q-values for all actions
        raise NotImplementedError
