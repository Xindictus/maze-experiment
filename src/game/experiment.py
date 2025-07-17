from dataclasses import dataclass
from typing import List

import numpy as np
import torch as T
from pydantic import BaseModel

from src.marl.algos.common import Observation


@dataclass
class Experiment:
    _config: BaseModel
    _global_obs: Observation = None

    @property
    def global_observation(self) -> np.ndarray:
        if self._global_obs is None:
            raise ValueError("Global observation not set.")
        return self._global_obs.get_state()

    @global_observation.setter
    def global_observation(self, val: List[float]):
        self._global_obs = Observation(
            config=self._config,
            normalized=self._normalize_global_state(val),
            raw_input=val,
        )

    def _normalize_feature(
        self, feat: float, min_v: float, max_v: float
    ) -> float:
        # Normalize features to range [-1, 1]
        val = min(max(feat, min_v), max_v)
        return 2 * (val - min_v) / (max_v - min_v) - 1

    def _normalize_global_state(self, observation: List[float]) -> np.ndarray:
        # Normalize observation features - 8 features expected
        norm_observation = [0] * len(observation)

        # Ball x & y position from -2/2 to -1/1
        norm_observation[0] = self._normalize_feature(observation[0], -2, 2)
        norm_observation[1] = self._normalize_feature(observation[1], -2, 2)

        # Ball x & y velocity from -0/2 to 0/1
        norm_observation[2] = self._normalize_feature(observation[2], -4, 4)
        norm_observation[3] = self._normalize_feature(observation[3], -4, 4)

        # Board angle f & t from -30/30 to -1/1
        norm_observation[4] = self._normalize_feature(observation[4], -30, 30)
        norm_observation[5] = self._normalize_feature(observation[5], -30, 30)

        # Board f & t velocity from -1/1 to -1/1
        norm_observation[6] = self._normalize_feature(
            observation[6], -1.9, 1.9
        )
        norm_observation[7] = self._normalize_feature(
            observation[7], -1.9, 1.9
        )

        return np.clip(norm_observation, -1.3, 1.3)

    def get_local_obs(self, agent_id: int) -> np.ndarray:
        if agent_id == 0:
            return self.global_observation.slice([0, 2, 4, 6])
        elif agent_id == 1:
            return self.global_observation.slice([1, 3, 5, 7])
        else:
            raise ValueError(f"Invalid agent ID: {agent_id}")

    def get_global_state_T(self) -> T.Tensor:
        return self._global_obs.to_tensor()
