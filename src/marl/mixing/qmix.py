import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.config.qmix_base import QmixBaseConfig
from src.utils.logger import Logger

logger = Logger().get_logger()


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

    def __init__(self, config: QmixBaseConfig):
        super(QMixer, self).__init__()

        logger.debug(config)
        self.config = config

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

    def forward(self, agent_qs, states):
        # TODO
        # - layer norm & batch norm
        # - weight clipping
        # - dropout in hypernetworks
        # logger.debug(f"{tensor.detach().cpu().numpy()}")
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        # also enforcing monotonicity?
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # elu activation
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        # also enforcing monotonicity?
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        # Qtot​(s,a)=fθ(Q1​,...,QN​)+V(s)
        return q_tot
