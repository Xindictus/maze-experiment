from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from config.reward_strategy import RewardStrategy


class SACBaseConfig(BaseModel):
    # Entropy regularization coefficient
    alpha: float = Field(default=0.0003, ge=0, le=1)

    # Experience buffer - batch size
    batch_size: int = Field(default=256, ge=1, le=4096)

    # Coefficient for entropy penalty or exploration regularization
    beta: float = Field(default=0.0003, ge=0, le=1)

    # Directory where agent checkpoints will be saved during training
    chkpt: str = Field(default=Path("rl_models/saved_models/"))

    # Action input
    discrete: bool = Field(default=True)

    # If True, the main agent's weights will not be updated during training
    freeze_agent: bool = Field(default=False)

    # If True, the second agent (if present) will be frozen during training
    freeze_second_agent: bool = Field(default=False)

    # Discount factor
    gamma: float = Field(default=0.99)

    # Number of variables in hidden layer 1
    layer1_size: int = Field(default=32, ge=4, le=1024)

    # Number of variables in hidden layer 2
    layer2_size: int = Field(default=32, ge=4, le=1024)

    # Learning rate
    learning_rate: float = Field(default=0.001, ge=1e-6, le=1)

    # Whether to load an existing checkpoint at the start of training
    load_checkpoint: bool = Field(default=True)

    # Path to the checkpoint file for the main agent
    load_file: str = Field(default=Path("rl_models/initial/"))

    # Whether to initialize and load a second agent
    load_second_agent: bool = Field(default=False)

    # Optional path to checkpoint for the second agent
    load_second_file: str | None = Field(default=None)

    # Identifier for saving model-related artifacts (used in logs, filenames)
    model_name: str = Field(default=Path("no_tl_participant"))

    # Scalar to adjust the automatic entropy tuning target
    target_entropy_ratio: float = Field(default=0.4, ge=0, le=1)

    # Soft update coefficient
    tau: float = Field(default=0.005, ge=0, le=1)

    """
    Type of reward function. Currently three reward functions are implemented:
    - Distance
    - Shafti
    - Timeout
    """
    reward_function: RewardStrategy = Field(default=RewardStrategy.SHAFTI)

    @field_validator("batch_size", "layer1_size", "layer2_size")
    @classmethod
    def must_be_power_of_two(cls, v: int, info) -> int:
        if (v & (v - 1)) != 0:
            raise ValueError(f"[{info.field_name}]: Must be power of 2!")
        return v
