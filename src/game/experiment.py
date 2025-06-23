import numpy as np

from typing import List


class Experiment:
    @staticmethod
    def normalize_feature(feat: float, min_v: float, max_v: float) -> float:
        # Normalize features to range [-1, 1]
        val = min(max(feat, min_v), max_v)
        return 2 * (val - min_v) / (max_v - min_v) - 1

    @staticmethod
    def normalize_state(observation: List[float]) -> np.ndarray:
        # Normalize observation features - 8 features expected

        norm_observation = [0] * len(observation)

        # x,y from -2/2 to -1/1
        norm_observation[0] = Experiment.normalize_feature(
            observation[0], -2, 2
        )
        norm_observation[1] = Experiment.normalize_feature(
            observation[1], -2, 2
        )

        # x,y velocity from -0/2 to 0/1
        norm_observation[2] = Experiment.normalize_feature(
            observation[2], -4, 4
        )
        norm_observation[3] = Experiment.normalize_feature(
            observation[3], -4, 4
        )

        # f,t from -30/30 to -1/1
        norm_observation[4] = Experiment.normalize_feature(
            observation[4], -30, 30
        )
        norm_observation[5] = Experiment.normalize_feature(
            observation[5], -30, 30
        )

        # # f,t velocity from -1/1 to -1/1
        norm_observation[6] = Experiment.normalize_feature(
            observation[6], -1.9, 1.9
        )
        norm_observation[7] = Experiment.normalize_feature(
            observation[7], -1.9, 1.9
        )

        return np.clip(norm_observation, -1.3, 1.3)
        # for i in range(len(norm_observation)):
        #     if norm_observation[i] > 1.3:
        #         norm_observation[i] = 1.3
        #     elif norm_observation[i] < -1.3:
        #         norm_observation[i] = -1.3

        # return np.array(norm_observation)
