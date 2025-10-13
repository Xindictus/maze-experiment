from typing import List, Literal, Optional

import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.common import ActionSpace, Observation
from src.marl.algos.qmix import QmixAgent, QmixQNetNetwork
from src.utils.logger import Logger


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
                name=f"Agent-[{i + 1:03d}]",
            )
            for i in range(config.n_agents)
        ]
        self.config = config

    def init_hidden(self) -> None:
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
        return self.agents[agent_id].forward(obs)

    def select_actions(
        self,
        observations: List[float],
        # env: Experiment,
        epsilon: float,
        mode: Literal["test", "train"] = "test",
    ) -> List[int]:
        """
        Selects agent actions using epsilon-greedy.

        Returns:
            List[int]: Selected actions, one per agent.
        """
        actions: List[int] = []

        for agent_id, agent in enumerate(self.agents):
            obs = Observation(
                config=self.config,
                normalized=observations[agent_id],
                # normalized=env.get_local_obs(agent_id),
            )

            if mode == "test":
                Logger().debug((agent_id, obs))

            action = agent.select_action(
                obs=obs.to_tensor(), epsilon=epsilon, mode=mode
            )

            actions.append(action)

        return actions

    def parameters(self) -> List[T.nn.Parameter]:
        return [p for agent in self.agents for p in agent.parameters()]

    @T.no_grad()
    def load_state(
        self,
        other: "MAC",
        update: Literal["hard", "soft"] = "hard",
        tau: Optional[float] = None,
    ) -> None:
        # Both MAC instances should have the same number of agents
        assert len(self.agents) == len(
            other.agents
        ), "MAC agent count mismatch during load_state"

        if update == "hard":
            for agent, other_agent in zip(self.agents, other.agents):
                agent.load_state(other_agent)
        elif update == "soft":
            for agent, other_agent in zip(self.agents, other.agents):
                for p_s, p_o in zip(
                    agent.network.parameters(),
                    other_agent.network.parameters(),
                ):
                    p_s.data.mul_(1 - tau).add_(p_o.data, alpha=tau)
