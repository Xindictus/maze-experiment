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


@pytest.mark.parametrize("layers", [1, 2])
def test_qmix_forward_output_shape(layers):
    config = QmixBaseConfig(
        n_agents=2,
        state_shape=(8,),
        embed_dim=32,
        hypernet_embed=64,
        hypernet_layers=layers,
    )

    _, mixer = create_test_mixer(config)

    assert mixer.config.state_dim == 8

    agent_qs, states = create_dummy_inputs()

    q_tot = mixer(agent_qs, states)

    assert q_tot.shape == (5, 200, 1)


@pytest.mark.parametrize("layers", [1, 2])
def test_qmix_forward_output_monotonicity(layers):
    config = QmixBaseConfig(
        n_agents=2,
        state_shape=(8,),
        embed_dim=32,
        hypernet_embed=64,
        hypernet_layers=layers,
    )

    _, mixer = create_test_mixer(config)

    assert mixer.config.state_dim == 8

    agent_qs, states = create_dummy_inputs()

    base_q_tot = mixer(agent_qs, states)

    assert torch.all(torch.isfinite(base_q_tot))
    assert base_q_tot.ndim == 3

    # Increase agent 0's Q-value slightly
    agent_qs[:, :, 0] += 0.1
    new_q_tot = mixer(agent_qs, states)

    # Q_tot should increase or stay the same
    assert (new_q_tot >= base_q_tot).all()


@pytest.mark.parametrize("layers", [1, 2])
def test_qmix_backward_pass(layers):
    config = QmixBaseConfig(
        n_agents=2,
        state_shape=(8,),
        embed_dim=32,
        hypernet_embed=64,
        hypernet_layers=layers,
    )

    _, mixer = create_test_mixer(config)

    agent_qs, states = create_dummy_inputs()

    q_tot = mixer(agent_qs, states)
    loss = q_tot.sum()
    loss.backward()

    # Check that parameters received gradients
    grads = [p.grad for p in mixer.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
