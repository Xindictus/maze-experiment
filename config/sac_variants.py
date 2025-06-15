from pathlib import Path
from pydantic import Field

from config.sac_base import SACBaseConfig


class DefaultConfig(SACBaseConfig):
    pass


class AgentAgentConfig(SACBaseConfig):
    # If True, the main agent's weights will not be updated during training
    freeze_agent: bool = Field(default=True)

    # Path to the checkpoint file for the main agent
    load_file: str = Field(default=Path("rl_models/policy_transfer/"))

    # Whether to initialize and load a second agent
    load_second_agent: bool = Field(default=True)

    # Optional path to checkpoint for the second agent
    load_second_file: str = Field(default=Path("rl_models/initial/"))


class AgentOnlyConfig(SACBaseConfig):
    # Experience buffer - batch size
    batch_size: int = Field(default=64, ge=1, le=4096)

    # If True, the main agent's weights will not be updated during training
    freeze_agent: bool = Field(default=True)

    # Whether to load an existing checkpoint at the start of training
    load_checkpoint: bool = Field(default=False)


class EvaluationConfig(SACBaseConfig):
    pass


class HumanDataCollectConfig(SACBaseConfig):
    pass
