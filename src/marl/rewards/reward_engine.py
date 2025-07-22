class RewardEngine:
    @staticmethod
    def compute_reward(reached_goal: bool, timed_out: bool) -> int:
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
        if reached_goal and not timed_out:
            return 10

        return -1
