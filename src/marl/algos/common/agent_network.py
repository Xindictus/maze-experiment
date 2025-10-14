from abc import ABC, abstractmethod
from typing import Optional

import torch as T
import torch.nn as nn
from pydantic import BaseModel


class AgentNetwork(nn.Module, ABC):
    def __init__(self, config: BaseModel):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, obs: T.Tensor, hidden: Optional[T.Tensor] = None
    ) -> T.Tensor:
        # Returns the Q-values for all actions
        raise NotImplementedError
