from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch as T
from pydantic import BaseModel


@dataclass
class Observation:
    config: BaseModel
    normalized: np.ndarray
    init_ball_pos: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    raw_input: Optional[List[float]] = None

    def get_state(self) -> np.ndarray:
        obs = self.normalized

        if self.config.is_extended_obs_enabled:
            obs = np.concatenate([obs, self.init_ball_pos])

        return obs

    def slice(self, indices: List[int]) -> np.ndarray:
        obs = self.normalized[indices]

        if self.config.is_extended_obs_enabled:
            obs = np.concatenate([obs, self.init_ball_pos])

        return obs

    def to_tensor(self) -> T.Tensor:
        return T.tensor(self.get_state(), dtype=T.float32).to(
            self.config.device
        )
