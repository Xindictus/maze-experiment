from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RewardContext:
    reached_goal: bool
    timed_out: bool
    dist_travelled: float = 0.0
    distance_from_goal: float = 0.0


@dataclass
class RewardEngine(ABC):
    goal_reward: float
    min_distance_delta: float
    reward_scale: float
    stall_penalty: float
    # number of steps to consider as stalling
    stall_threshold: int
    timeout_penalty: float

    @abstractmethod
    def compute_reward(self, ctx: RewardContext) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
