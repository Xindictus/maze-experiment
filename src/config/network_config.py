from pydantic import BaseModel, Field, field_validator

from src.config.validators import must_be_valid_url


class NetworkConfig(BaseModel):
    # Entropy regularization coefficient
    ip_distributor: str = Field(default="http://localhost:8080")

    # Experience buffer - batch size
    maze_rl: str = Field(default="http://localhost:8080")

    # Coefficient for entropy penalty or exploration regularization
    maze_server: str = Field(default="http://localhost:8080")

    # Restrict values to power of 2
    _check_url = field_validator("ip_distributor", "maze_rl", "maze_server")(
        must_be_valid_url
    )
