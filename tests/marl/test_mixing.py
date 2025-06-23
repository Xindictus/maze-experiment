import torch
import torch.nn as nn
import pytest

from src.marl.mixing.qmix import QMixer
from src.config.qmix_base import QmixBaseConfig


def create_test_mixer(config: QmixBaseConfig = None) -> QMixer:
    if config is None:
        config = QmixBaseConfig(
            n_agents=2,
            state_shape=(5, 5, 3),
            embed_dim=32,
            hypernet_embed=64,
            hypernet_layers=1,
        )

    return (config, QMixer(config))


def create_dummy_inputs(
    batch_size: int = 5,
    episode_len: int = 200,
    n_agents: int = 2,
    state_dim: int = 8,
):
    agent_qs = torch.rand(batch_size, episode_len, n_agents)
    states = torch.rand(batch_size, episode_len, state_dim)

    return (agent_qs, states)


def get_out_feat(layer: nn.Module) -> int:
    if isinstance(layer, nn.Sequential):
        return layer[-1].out_features
    return layer.out_features


@pytest.mark.parametrize("layers", [1, 2])
def test_qmixer_constructor(layers):
    config = QmixBaseConfig(
        n_agents=3,
        state_shape=(5, 5, 3),
        embed_dim=32,
        hypernet_embed=64,
        hypernet_layers=layers,
    )

    _, mixer = create_test_mixer(config)

    # Check dimensions
    assert mixer.config.n_agents == 3
    assert mixer.config.state_dim == 75
    assert get_out_feat(mixer.hyper_w_1) == 3 * 32
    assert get_out_feat(mixer.hyper_w_final) == config.embed_dim
    assert get_out_feat(mixer.hyper_b_1) == config.embed_dim
    assert get_out_feat(mixer.V) == 1

    assert isinstance(mixer.V, nn.Sequential)


def test_qmix_forward_output_shape():
    config = QmixBaseConfig(
        n_agents=2,
        state_shape=(8,),
        embed_dim=32,
        hypernet_embed=64,
        hypernet_layers=1,
    )

    _, mixer = create_test_mixer(config)

    assert mixer.config.state_dim == 8

    agent_qs, states = create_dummy_inputs()

    q_tot = mixer(agent_qs, states)

    assert q_tot.shape == (5, 200, 1)


def test_qmix_forward_output_monotonicity():
    pass
