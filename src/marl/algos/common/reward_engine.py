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
    goal_reward: int
    reward_scale: int
    timeout_penalty: int

    @abstractmethod
    def compute_reward(ctx: RewardContext) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
