from typing import List, Literal, Tuple

from numpy import prod
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config.validators import must_be_power_of_two


class QmixBaseConfig(BaseModel):
    device: str = Field(default="cuda")

    # Final output dimension of the mixer hidden layer
    embed_dim: int = Field(32, gt=0)

    # List of hidden layer sizes for the agent network (e.g., [64, 64])
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 32])

    # Hidden layer size in the hypernet (if 2-layer MLP is used)
    hypernet_embed: int = Field(64, gt=0)

    """
    Number of layers in each hypernetwork:
    - 1 for linear
    - 2 for MLP
    """
    hypernet_layers: Literal[1, 2] = Field(default=1)

    # The slice of the global observation (local observation)
    input_dim: int = Field(default=4)

    # Number of actions
    n_actions: int = Field(default=3)

    # Number of agents
    n_agents: int = Field(default=2, ge=2, le=100)

    #  Shape of the global state tensor
    state_shape: Tuple[int, ...] = Field(default=(4,), min_length=1)

    # Restrict values to power of 2
    # TODO: hidden dim
    _check_power = field_validator("embed_dim", "hypernet_embed")(
        must_be_power_of_two
    )

    @model_validator(mode="after")
    def compute_flattened_state_dim(self) -> "QmixBaseConfig":
        object.__setattr__(self, "state_dim", int(prod(self.state_shape)))

        return self
