from typing import Literal

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.config.qmix_base import QmixBaseConfig
from src.utils.logger import Logger


class QMixer(nn.Module):
    """
    Initializes a QMIX mixing network.

    The network dynamically generates weights to combine
    individual agent Q-values into a monotonic joint Q_tot value.

    All weight and bias parameters are conditioned on
    the global state via hypernetworks.

    The architecture of the hypernetworks is controlled via `hypernet_layers`:
    - 1: Linear
    - 2: MLP (Linear -> ReLU -> Linear)
    """

    def __init__(
        self,
        config: QmixBaseConfig,
        name: Literal["MAIN", "TARGET"],
        buffer_type: Literal["episode", "standard"] = "episode",
    ):
        super(QMixer, self).__init__()

        Logger().debug(config)
        self.config = config
        self.name = name
        self.buffer_type = buffer_type

        """
        ### Hypernet Layers == 1
        Generate first-layer weights and final weights
        via state-conditioned hypernetworks.
        Linear or MLP depending on hypernet_layers

        ### Hypernet Layers == 2
        -- hypernetworks p.6-7,13--

        Producing weights of appropriate size + the
        final bias of the mixing networks.

        (hidden layer of 32 units with RELU non-linearity)
        """
        self.hyper_w_1 = self._create_hypernet(
            config.state_dim,
            config.embed_dim * config.n_agents,
        )
        self.hyper_w_final = self._create_hypernet(
            config.state_dim,
            config.embed_dim,
        )

        """
        State dependent bias for hidden layer.
        Hypernet for the biases in the first mixing layer.
        """
        self.hyper_b_1 = nn.Linear(config.state_dim, config.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(config.state_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 1),
        )

    def _create_hypernet(self, input_dim: int, output_dim: int) -> nn.Module:
        hyper_layers = self.config.hypernet_layers
        match hyper_layers:
            case 1:
                return nn.Linear(input_dim, output_dim)
            case 2:
                hidden = self.config.hypernet_embed
                return nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, output_dim),
                )
            case _:
                raise ValueError(
                    f"Unsupported hypernet_layers: {hyper_layers}"
                )

    def _is_episode_buffer(self) -> bool:
        return self.buffer_type == "episode"

    def _is_standard_buffer(self) -> bool:
        return self.buffer_type == "standard"

    def forward(self, agent_qs: T.Tensor, states: T.Tensor) -> T.Tensor:
        Logger().debug(
            f"[{self.name}] agent_qs (shape.before): {agent_qs.shape}"
        )
        Logger().debug(f"[{self.name}] states (shape.before): {states.shape}")
        Logger().debug(
            f"[{self.name}] state_dim config (shape.before): {self.config.state_dim}"
        )

        # TODO
        # - layer norm & batch norm
        # - weight clipping
        # - dropout in hypernetworks

        """
        States has the shape [batch_size, episode_length, state_dim]

        In our case, batch size is the number of games in each block (5) that
        we are going to process in parallel as part of this training batch,
        episode length is the number of timesteps per episode (40s for each
        game, exchanging actions/states every 200ms, which equals to 200
        timesteps) and finally state dimensionality, which is the output of
        `normalize_state` (8,).

        We are reshaping states, because networks expect 2D input instead
        of our current 3D tensor.
        """
        Logger().debug(f"[{self.name}] states: {states}")
        Logger().debug(f"[{self.name}] agent_qs: {agent_qs}")

        B, Tq, N = agent_qs.shape

        # TODO: Assumes that episode stride is always 1.
        #       Will need to adjust this to accommodate for strides > 1.
        if self.name == "MAIN":
            states = states[:, :Tq, :]

            if self._is_standard_buffer():
                agent_qs = agent_qs[:, :Tq, :]
        elif self.name == "TARGET":
            states = states[:, 1 : Tq + 1, :]

            if self._is_standard_buffer():
                agent_qs = agent_qs[:, 1 : Tq + 1, :]
        else:
            raise ValueError("Invalid mixer name")

        Logger().debug(f"[{self.name}] states (shape.slice): {states.shape}")
        Logger().debug(
            f"[{self.name}] agent_qs (agent_qs.slice): {agent_qs.shape}"
        )

        Tq = agent_qs.shape[1]
        states = states.reshape(B * states.shape[1], -1)
        agent_qs = agent_qs.reshape(B * Tq, 1, N)

        Logger().debug(
            f"[{self.name}] agent_qs (agent_qs.after): {agent_qs.shape}"
        )
        Logger().debug(f"[{self.name}] states (shape.after): {states.shape}")
        Logger().debug(f"[{self.name}] states: {states}")

        # ---------------- First layer ---------------- #

        """
        Generate the first layer mixing weights and biases
        based on the global state.
        """

        # Monotonicity is enforced here.
        w1 = T.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.config.n_agents, self.config.embed_dim)
        b1 = b1.view(-1, 1, self.config.embed_dim)

        Logger().debug(f"[{self.name}] w1 (shape): {w1.shape}")
        Logger().debug(f"[{self.name}] b1 (shape): {b1.shape}")

        # ELU activation
        hidden = F.elu(T.bmm(agent_qs, w1) + b1)

        # ---------------- Second layer ---------------- #

        # Generating final-layer weights. Monotonicity is enforced again.
        w_final = T.abs(self.hyper_w_final(states))

        # Reshaping to prepare for batched matrix multiplication.
        w_final = w_final.view(-1, self.config.embed_dim, 1)

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        y = T.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(B, Tq, 1)

        # Qtot​(s,a)= fθ(Q1​, ..., QN​) + V(s)
        return q_tot
