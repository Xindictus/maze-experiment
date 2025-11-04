from dataclasses import dataclass
from typing import Dict, List

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
    def global_observation(self, val: Dict[List[float], List[int]]):
        # val contains both the state and the initial ball position
        self._global_obs = Observation(
            config=self._config,
            normalized=self._normalize_global_state(val[0]),
            init_ball_pos=val[1],
            raw_input=val[0],
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

        # Ball z & x position
        norm_observation[0] = self._normalize_feature(observation[0], -2, 2)
        norm_observation[1] = self._normalize_feature(observation[1], -2, 2)

        # Ball z & x velocity
        norm_observation[2] = self._normalize_feature(observation[2], -4, 4)
        norm_observation[3] = self._normalize_feature(observation[3], -4, 4)

        # Board angle z & x
        norm_observation[4] = self._normalize_feature(observation[4], -30, 30)
        norm_observation[5] = self._normalize_feature(observation[5], -30, 30)

        # Board z & x angular speed
        norm_observation[6] = self._normalize_feature(
            observation[6], -1.9, 1.9
        )
        norm_observation[7] = self._normalize_feature(
            observation[7], -1.9, 1.9
        )

        return np.clip(norm_observation, -1.3, 1.3)

    def get_local_obs(self, agent_id: int) -> np.ndarray:
        common_obs = np.array([], dtype=float)
        obs_slices = {
            0: [0, 2, 4, 6],
            1: [1, 3, 5, 7],
        }

        # Ball position & velocity are common and not something agents control
        if self._config.is_common_obs_enabled:
            common_obs = self._global_obs.slice([0, 1, 2, 3])
            obs_slices = {
                0: [4, 6],
                1: [5, 7],
            }

        # Each agent gets at least its own board angle & angular speed
        if agent_id not in obs_slices:
            raise ValueError(f"Invalid agent ID: {agent_id}")

        return np.concatenate(
            [common_obs, self._global_obs.slice(obs_slices[agent_id])]
        )

    def get_global_state_T(self) -> T.Tensor:
        return self._global_obs.to_tensor()

    def get_env_actions(self, actions: List[int]) -> List[int]:
        """_summary_

        Args:
            actions (List[int]): Agent actions

        Returns:
            List[int]: Env compatible actions
        """
        return [-1 if action == 2 else action for action in actions]
