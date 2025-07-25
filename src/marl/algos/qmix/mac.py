from typing import List

import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.game.experiment import Experiment
from src.marl.algos.common import ActionSpace, Observation
from src.marl.algos.qmix import QmixAgent, QmixQNetNetwork


class MAC:
    def __init__(
        self,
        config: QmixBaseConfig,
    ):
        # Initialize agents
        self.agents: List[QmixAgent] = [
            QmixAgent(
                action_space=ActionSpace(list(range(3))),
                config=config,
                network=QmixQNetNetwork(config=config),
            )
            for _ in range(config.n_agents)
        ]
        self.config = config

    def init_hidden(self):
        """
        Initializes hidden states for all agents - GRU only
        """
        pass
        # for agent in self.agents:
        #     agent.network.init_hidden(batch size ??)

    def forward(self, agent_id: int, obs: T.Tensor) -> T.Tensor:
        """
        Forward pass for a single agent.
        """
        return self.agents[agent_id].network(obs)

    def select_actions(self, env: Experiment, epsilon: float) -> List[int]:
        """
        Selects agent actions using epsilon-greedy.

        Returns:
            List[int]: Selected actions, one per agent.
        """
        actions: List[int] = []

        for agent_id, agent in enumerate(self.agents):
            obs = Observation(
                config=self.config,
                normalized=env.get_local_obs(agent_id),
            )
            action = agent.select_action(obs.to_tensor(), epsilon)
            actions.append(action)

        return actions

    def parameters(self) -> List[T.nn.Parameter]:
        return [p for agent in self.agents for p in agent.parameters()]

    def load_state(self, other: "MAC"):
        # Both MAC instances should have the same number of agents
        assert len(self.agents) == len(
            other.agents
        ), "MAC agent count mismatch during load_state"

        for agent, target_agent in zip(self.agents, other.agents):
            target_agent.load_state(agent)
