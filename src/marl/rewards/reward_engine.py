from src.utils.logger import Logger


class RewardEngine:
    @staticmethod
    def compute_reward(
        reached_goal: bool, timed_out: bool, dist_travelled: float = 0.0
    ) -> int:
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
        logger = Logger().with_context("RewardEngine")
        logger.debug(f"Distance travelled: {dist_travelled:0.2f}")

        if reached_goal and not timed_out:
            return 10

        return -1
