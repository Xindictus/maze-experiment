from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field


class GameBaseConfig(BaseModel):
    # True if playing only the RL agent (no human-in-the-loop)
    agent_only: bool = Field(default=False)

    # Input speed as translated by the game
    agent_speed: int = Field(default=50, ge=25, le=100)

    """
    - Date and time of the experiments.
    - Used loading the model created that date (if asked by the user)
    """
    checkpoint_name: Path = Field(
        default=Path(f'{datetime.now().strftime("%Y%m%d%H%M%S")}')
    )

    # Position of the goal on the board
    discrete_angle_change: int = Field(default=3)

    # True for Discrete or False for Continuous human input
    discrete_input: bool = Field(default=False)

    # "left_down" "left_up" "right_down"
    goal: str = Field(default="left_down")

    # TODO
    human_assist: bool = Field(default=True)

    # TODO
    human_only: bool = Field(default=False)

    # Input speed as translated by the game
    human_speed: int = Field(default=50, ge=25, le=100)

    # True if loading stored model
    load_checkpoint: bool = Field(default=False)

    # Save models and logs
    save: bool = Field(default=True)

    # False if playing with RL agent
    second_human: bool = Field(default=False)

    # True if no training happens
    test_model: bool = Field(default=False)

    # Used for logging
    verbose: bool = Field(default=True)
