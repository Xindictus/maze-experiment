from typing import Literal

from pydantic import BaseModel, Field


class ExperimentBaseConfig(BaseModel):
    # Time duration in sec between consecutive RL agent actions
    action_duration: float = Field(default=0.2, ge=0.1, le=1)

    # The size of the buffer to initialize with
    buffer_memory_size: int = Field(default=1e6)

    # The type of buffer to be used
    buffer_type: Literal["episode", "prioritized", "standard"] = Field(
        default="episode"
    )

    # Number of rounds/games per block
    games_per_block: int = Field(default=5)

    # Goal zone radius
    goal_zone_radius: float = Field(default=0.05)

    # Reward for reaching the goal
    goal_reward: float = Field(default=10)

    # Max training games per experiment
    max_blocks: int = Field(default=5)

    # Max duration per game in seconds
    max_duration: int = Field(default=40)

    # Minimum distance delta to consider as progress
    min_distance_delta: float = Field(default=0.01)

    # Selection of reward engine
    reward_engine: Literal[
        "simple",
        "goal_distance",
        "progress_distance",
        "progress_with_stalling",
        "speed_stalling",
    ] = Field(default="goal_distance")

    # Scale for reward when not reaching goal
    reward_scale: float = Field(default=-0.1)

    # Scale for speed penalty
    speed_scale: float = Field(default=0.2)

    # Speed threshold below which penalty is applied
    speed_threshold: float = Field(default=0.8)

    # Penalty for stalling
    stall_penalty: float = Field(default=-1.0)

    # Number of consecutive steps with no progress to consider as  stalling
    stall_threshold: int = Field(default=8)

    # Penalty for timing out
    timed_out_penalty: float = Field(default=-1)

    # Number of training epochs
    update_cycles: int = Field(default=200)
