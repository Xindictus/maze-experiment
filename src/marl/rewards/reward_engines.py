from typing import Callable

from src.config.experiment_base import ExperimentBaseConfig
from src.marl.algos.common.reward_engine import RewardContext, RewardEngine
from src.utils.logger import Logger


class SimpleRewardEngine(RewardEngine):
    def compute_reward(self, ctx: RewardContext) -> float:
        logger = Logger().with_context("SimpleRewardEngine")
        logger.debug(f"Distance from goal: {ctx.distance_from_goal}")

        if ctx.reached_goal and not ctx.timed_out:
            return self.goal_reward

        return self.timeout_penalty

    def reset(self) -> None:
        pass


class GoalDistanceRewardEngine(RewardEngine):
    def compute_reward(self, ctx: RewardContext) -> float:
        logger = Logger().with_context("GoalDistanceRewardEngine")
        logger.debug(f"Distance from goal: {ctx.distance_from_goal}")

        if ctx.reached_goal and not ctx.timed_out:
            return self.goal_reward

        if ctx.timed_out:
            return self.timeout_penalty

        return self.reward_scale * abs(ctx.distance_from_goal)

    def reset(self) -> None:
        pass


class ProgressDistanceRewardEngine(RewardEngine):
    prev_distance: float = None

    def compute_reward(self, ctx: RewardContext) -> float:
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

    def reset(self) -> None:
        self.prev_distance = None


class ProgressWithStallingRewardEngine(RewardEngine):
    prev_distance: float = None
    stall_counter: int = 0

    def compute_reward(self, ctx: RewardContext) -> float:
        logger = Logger().with_context("ProgressWithStallingRewardEngine")
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

        raw_delta = self.prev_distance - ctx.distance_from_goal

        if raw_delta < self.min_distance_delta:
            self.stall_counter += 1
            logger.debug(
                f"Stall counter increased to {self.stall_counter} "
                f"(raw_delta: {raw_delta})"
            )
        else:
            logger.debug(f"Stall counter reset to 0 (raw_delta: {raw_delta})")
            self.stall_counter = 0

        delta = raw_delta
        extra_penalty = 0.0

        if self.stall_counter >= self.stall_threshold:
            extra_penalty = self.stall_penalty
            logger.debug(
                f"Stall threshold reached ({self.stall_counter} >= {self.stall_threshold}). "
                f"Applying stall penalty {extra_penalty}."
            )
            # reset counter so we don't re-penalize every single step
            self.stall_counter = 0
            delta = 0.0

        # Updates for next step
        self.prev_distance = ctx.distance_from_goal

        progress_reward = self.reward_scale * delta

        total_reward = progress_reward + extra_penalty

        logger.debug(
            f"raw_delta={raw_delta}, delta_used={delta}, "
            f"shaped_reward={progress_reward}, extra_penalty={extra_penalty}, "
            f"total_reward={total_reward}"
        )

        return total_reward

    def reset(self) -> None:
        self.prev_distance = None
        self.stall_counter = 0


class SpeedStallingRewardEngine(ProgressWithStallingRewardEngine):
    def compute_reward(self, ctx: RewardContext) -> float:
        logger = Logger().with_context("SpeedStallingRewardEngine")
        logger.debug(
            f"Prev distance: {self.prev_distance} | "
            f"Distance from goal: {ctx.distance_from_goal} | "
            f"Ball speed: {ctx.ball_speed}"
        )

        total_reward = super().compute_reward(ctx)

        if ctx.distance_from_goal <= self.goal_zone_radius:
            effective_speed = max(0.0, ctx.ball_speed - self.speed_threshold)
            speed_penalty = -self.speed_scale * effective_speed
        else:
            speed_penalty = 0.0

        total_reward += speed_penalty

        logger.debug(
            f"speed_penalty={speed_penalty} | total_reward={total_reward}"
        )

        return total_reward


type ExportFN = Callable[[str], RewardEngine]

# Use registry pattern to get reward engine
reward_engines: dict[str, ExportFN] = {
    "simple": SimpleRewardEngine,
    "goal_distance": GoalDistanceRewardEngine,
    "progress_distance": ProgressDistanceRewardEngine,
    "progress_with_stalling": ProgressWithStallingRewardEngine,
    "speed_stalling": SpeedStallingRewardEngine,
}


def get_reward_engine(name: str, config: ExperimentBaseConfig) -> RewardEngine:
    engine = reward_engines.get(name)

    if engine is None:
        raise ValueError(f"Unknown reward engine: {name}")

    return engine(
        goal_zone_radius=config.goal_zone_radius,
        goal_reward=config.goal_reward,
        min_distance_delta=config.min_distance_delta,
        reward_scale=config.reward_scale,
        speed_scale=config.speed_scale,
        speed_threshold=config.speed_threshold,
        stall_penalty=config.stall_penalty,
        stall_threshold=config.stall_threshold,
        timeout_penalty=config.timed_out_penalty,
    )
