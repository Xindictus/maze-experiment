from pydantic import Field

from src.config.game_base import GameBaseConfig


class DefaultConfig(GameBaseConfig):
    pass


class AgentAgentConfig(GameBaseConfig):
    pass


class AgentOnlyConfig(GameBaseConfig):
    # True if playing only the RL agent (no human-in-the-loop)
    agent_only: bool = Field(default=True)
    # TODO
    human_assist: bool = Field(default=False)


class EvaluationConfig(GameBaseConfig):
    pass


class HumanDataCollectConfig(GameBaseConfig):
    # TODO
    human_only: bool = Field(default=True)
