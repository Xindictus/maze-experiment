from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class GameBaseConfig(BaseModel):
    # True if playing only the RL agent (no human-in-the-loop)
    # TODO: Unused
    agent_only: bool = Field(default=False)

    # Input speed as translated by the game
    # TODO: Unused
    agent_speed: int = Field(default=50, ge=25, le=100)

    """
    - Date and time of the experiments.
    - Used loading the model created that date (if asked by the user)
    """
    # TODO: Unused
    checkpoint_name: Path = Field(
        default=Path(f'{datetime.now().strftime("%Y%m%d%H%M%S")}')
    )

    # Position of the goal on the board
    # TODO: Unused
    discrete_angle_change: int = Field(default=3)

    # True for Discrete or False for Continuous human input
    # TODO: Unused
    discrete_input: bool = Field(default=False)

    # left_down|left_up|right_down
    goal: str = Field(default="left_down")

    # TODO: Unused
    human_assist: bool = Field(default=True)

    # TODO: Unused
    human_only: bool = Field(default=False)

    # Input speed as translated by the game
    # TODO: Unused
    human_speed: int = Field(default=50, ge=25, le=100)

    # Save models and logs
    # TODO: Unused
    save: bool = Field(default=True)

    # False if playing with RL agent
    # TODO: Unused
    second_human: bool = Field(default=False)

    # True if no training happens
    # TODO: Unused
    test_model: bool = Field(default=False)
