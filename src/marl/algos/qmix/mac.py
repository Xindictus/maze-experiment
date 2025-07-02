import torch as T

from typing import List

from src.config.qmix_base import QmixBaseConfig
from src.game.experiment import Experiment
from src.marl.algos.common import Observation
from src.marl.algos.qmix import QmixAgent, QmixTrainer
from src.marl.mixing.qmix import QMixer


class MAC:
    def __init__(
        self,
        agents: List[QmixAgent],
        config: QmixBaseConfig,
        mixer: QMixer,
        trainer: QmixTrainer,
    ):
        self.agents = agents
        self.config = config
        self.experiment = Experiment
        self.mixer = mixer
        self.trainer = trainer

    def init_hidden(self):
        """
        Initializes hidden states for all agents - GRU only
        """
        # for agent in self.agents:
        #     agent.network.init_hidden()

    def forward(self, epsilon: float) -> List[int]:
        """
        Performs forward pass for all agents and
        selects actions using epsilon-greedy.

        Returns:
            List[int]: Selected actions, one per agent.
        """
        actions: List[int] = []

        for agent_id, agent in enumerate(self.agents):
            local_obs = self.experiment.get_local_obs(agent_id=agent_id)
            obs = Observation(config=self.config, normalized=local_obs)
            action = agent.select_action(obs.to_tensor(), epsilon)
            actions.append(action)

    def compute_q_tot(self, agent_qs):
        """_summary_

        Args:
            agent_qs (_type_): _description_
        """
        # TODO: Pass individual Qs in the mixer
        agent_qs_stack = T.stack(agent_qs, dim=1)
        global_state = self.experiment.get_global_state_T()
        self.mixer(agent_qs_stack, global_state)

    def on_batch_train(self, batch):
        # TODO: Trainer
        pass
