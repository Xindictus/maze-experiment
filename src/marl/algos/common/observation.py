import torch as T
import numpy as np

from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Optional


@dataclass
class Observation:
    config: BaseModel
    normalized: np.ndarray
    raw_input: Optional[List[float]] = None

    def get_state(self):
        return self.normalized

    def slice(self, indices: List[int]) -> np.ndarray:
        return self.normalized[indices]

    def to_tensor(self) -> T.Tensor:
        return T.tensor(self.get_state(), dtype=T.float32).to(
            self.config.device
        )
