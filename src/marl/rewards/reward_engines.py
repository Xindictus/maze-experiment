from src.marl.algos.common.reward_engine import RewardContext, RewardEngine
from src.utils.logger import Logger


class SimpleRewardEngine(RewardEngine):
    @staticmethod
    def compute_reward(ctx: RewardContext) -> int:
        """
        Computes reward based on goal achievement or timeout.

        - Each timestep: -1
        - Goal reached: +10
        - Timeout: -1 (no extra penalty here, unless you want to)

        Args:
            goal_reached (bool)
            timed_out (bool)

        Returns:
            int: reward value
        """
        logger = Logger().with_context("SimpleRewardEngine")
        logger.debug(f"Distance travelled: {ctx.dist_travelled:0.2f}")

        if ctx.reached_goal and not ctx.timed_out:
            return 10

        return -1


class GoalDistanceRewardEngine(RewardEngine):
    @staticmethod
    def compute_reward(ctx: RewardContext) -> int:
        logger = Logger().with_context("GoalDistanceRewardEngine")
        logger.debug(f"Context: {ctx}")

        if ctx.reached_goal and not ctx.timed_out:
            return 10

        if ctx.timed_out:
            return -1

        return -0.1 * abs(ctx.distance_from_goal)
