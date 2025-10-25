from typing import Callable

from src.config.experiment_base import ExperimentBaseConfig
from src.marl.algos.common.reward_engine import RewardContext, RewardEngine
from src.utils.logger import Logger


class SimpleRewardEngine(RewardEngine):
    def compute_reward(self, ctx: RewardContext) -> int:
        logger = Logger().with_context("SimpleRewardEngine")
        logger.debug(f"Distance from goal: {ctx.distance_from_goal}")

        if ctx.reached_goal and not ctx.timed_out:
            return self.goal_reward

        return self.timeout_penalty


class GoalDistanceRewardEngine(RewardEngine):
    def compute_reward(self, ctx: RewardContext) -> int:
        logger = Logger().with_context("GoalDistanceRewardEngine")
        logger.debug(f"Distance from goal: {ctx.distance_from_goal}")

        if ctx.reached_goal and not ctx.timed_out:
            return self.goal_reward

        if ctx.timed_out:
            return self.timeout_penalty

        return self.reward_scale * abs(ctx.distance_from_goal)


class ProgressDistanceRewardEngine(RewardEngine):
    prev_distance: float = None

    def compute_reward(self, ctx: RewardContext) -> int:
        logger = Logger().with_context("ProgressDistanceRewardEngine")
        logger.debug(
            f"Prev distance: {self.prev_distance} | "
            f"Distance from goal: {ctx.distance_from_goal}"
        )

        if ctx.reached_goal and not ctx.timed_out:
            return self.goal_reward

        if ctx.timed_out:
            return self.timeout_penalty

        if self.prev_distance is None:
            self.prev_distance = ctx.distance_from_goal

        delta = self.prev_distance - ctx.distance_from_goal
        self.prev_distance = ctx.distance_from_goal

        return self.reward_scale * delta


type ExportFN = Callable[[str], RewardEngine]

# Use registry pattern to get reward engine
reward_engines: dict[str, ExportFN] = {
    "simple": SimpleRewardEngine,
    "goal_distance": GoalDistanceRewardEngine,
    "progress_distance": ProgressDistanceRewardEngine,
}


def get_reward_engine(name: str, config: ExperimentBaseConfig) -> RewardEngine:
    engine = reward_engines.get(name)

    if engine is None:
        raise ValueError(f"Unknown reward engine: {name}")

    return engine(
        goal_reward=config.goal_reward,
        reward_scale=config.reward_scale,
        timeout_penalty=config.timed_out_penalty,
    )
