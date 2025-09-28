from pydantic import BaseModel, Field


class ExperimentBaseConfig(BaseModel):
    # Time duration in sec between consecutive RL agent actions
    action_duration: float = Field(default=0.2, ge=0.1, le=1)

    # TODO: Currently unused, possibly not needed
    agent: str = Field(default="sac")

    # The size of the buffer to initialize with
    buffer_memory_size: int = Field(default=1e6)

    # TODO: update_cycles
    # Number of train epochs
    epochs: int = Field(default=200)
    # epochs: int = Field(default=4000)

    # Number of rounds/games per block
    games_per_block: int = Field(default=10)

    # Perform offline gradient updates after every n episodes
    # TODO: Unused
    learn_every_n_games: int = Field(default=10)

    # Print avg reward in the interval
    log_interval: int = Field(default=10)

    # Max training games per experiment
    max_blocks: int = Field(default=6)

    # Max duration per game in seconds
    max_duration: int = Field(default=40)

    """
    max_games_mode: Experiment iterates over games.
                    Each game has a fixed max duration in seconds.
    max_interactions_mode: Experiment iterates over steps.
                           Each game has a fixed max duration in seconds.
    """
    # Choose max_games_mode or max_interactions_mode
    # TODO: Unused
    mode: str = Field(default="no_tl")

    # True if a single gradient update happens after every state transition
    # TODO: Unused
    online_updates: bool = Field(default=False)

    # TODO: Unused
    reward_scale: int = Field(default=2)

    """
    Offline gradient updates allocation
    Normal: allocates evenly the total number of updates through each session
    Descending: allocation of total updates using geometric progression
                with ratio 1/2
    descending normal big_first
    """
    scheduling: str = Field(default="normal")

    # True to start experiment with testing human with random agent
    # TODO: Unused
    start_with_testing_random_agent: bool = Field(default=True)

    # TODO: Unused
    test_interval: int = Field(default=10)

    # Total number of offline gradient updates throughout the whole experiment
    # TODO: Unused
    updates_per_ogu: int = Field(default=250)
