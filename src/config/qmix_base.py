from typing import ClassVar, List, Literal, Tuple

from numpy import prod
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config.validators import must_be_power_of_two


class QmixBaseConfig(BaseModel):
    PROCESSED: ClassVar[bool] = False

    # The type of agent network to be used
    agent_network_type: Literal["qnet", "gru"] = Field(default="qnet")

    # The number of transition states saved in each episode
    batch_episode_size: int = Field(default=16, ge=2, le=4096)

    # The sample sizes to pick each time
    batch_size: int = Field(default=256, ge=2, le=4096)

    # Default device to be used
    # TODO: Restrict to cuda/cpu choices
    device: str = Field(default="cuda")

    # Final output dimension of the mixer hidden layer
    embed_dim: int = Field(default=32, gt=0)

    # The decay rate factor for epsilon-greedy exploration
    # TODO-DEPRECATE
    epsilon_decay_rate: float = Field(default=0.05, ge=0.0)

    # The method to use for epsilon decay
    epsilon_decay_method: Literal[
        "cosine",
        "exp_half_life",
        "inverse_time",
        "linear",
        "logistic",
        "original",
        "polynomial",
    ] = Field(default="linear")

    # Discount factor for future rewards.
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    """
    Global norm to clip gradients during backprop.
    Prevents exploding gradients.
    """
    grad_norm_clip: float = Field(default=10.0, ge=0.0)

    # List of hidden layer sizes for the agent network (e.g., [64, 64])
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 32])

    # Hidden layer size in the hypernet (if 2-layer MLP is used)
    hypernet_embed: int = Field(default=64, gt=0)

    """
    Number of layers in each hypernetwork:
    - 1 for linear
    - 2 for MLP
    """
    hypernet_layers: Literal[1, 2] = Field(default=1)

    # The slice of the global observation (local observation)
    input_dim: int = Field(default=4)

    # Switch on/off for clipping
    is_grad_norm_clip_enabled: bool = Field(default=False)

    # Utilizes the initial ball position as part of the global/local state
    is_extended_obs_enabled: bool = Field(default=False)

    # Learning rate
    learning_rate: float = Field(default=0.0003, ge=1e-6, le=1)

    # Epsilon-greedy exploration strategy parameters
    max_epsilon: float = Field(default=1)
    min_epsilon: float = Field(default=0.01)

    # Number of actions
    n_actions: int = Field(default=3)

    # Number of agents
    n_agents: int = Field(default=2, ge=2, le=100)

    # Optimizer selection
    optimizer: Literal["adam", "adamw", "rms"] = Field(default="adam")

    # RMSProp smoothing constant (alpha).
    optim_alpha: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Controls the moving average of squared gradients.",
    )

    """
    Epsilon value for optimizer.
    """
    optim_eps: float = Field(default=1e-7, ge=0.0)

    #  Shape of the global state tensor
    state_shape: Tuple[int, ...] = Field(default=(8,), min_length=1)

    """
    Number of training steps between target network updates.
    Higher values slow updates for stability.
    """
    target_update_interval: int = Field(default=2, ge=1)

    # Controls how the agent target networks are updated
    target_update_mode: Literal["hard", "soft"] = Field(default="hard")

    # Soft update coefficient
    tau: float = Field(default=0.005, ge=0, le=1)

    # Restrict values to power of 2
    # TODO: hidden dim
    _check_power = field_validator(
        "batch_size",
        "embed_dim",
        "hypernet_embed",
        # "batch_episode_size", "batch_size", "embed_dim", "hypernet_embed"
    )(must_be_power_of_two)

    @model_validator(mode="after")
    def bump_obs_dim(self) -> "QmixBaseConfig":
        if not QmixBaseConfig.PROCESSED and self.is_extended_obs_enabled:
            self.input_dim += 3
            self.state_shape = (self.state_shape[0] + 3,)
            QmixBaseConfig.PROCESSED = True
        return self

    @model_validator(mode="after")
    def compute_flattened_state_dim(self) -> "QmixBaseConfig":
        object.__setattr__(self, "state_dim", int(prod(self.state_shape)))

        return self
