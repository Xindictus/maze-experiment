import torch.nn as nn
import pytest

from src.marl.mixing.qmix import QMixer
from src.config.qmix_base import QmixBaseConfig


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
    mixer = QMixer(config)

    # Check dimensions
    assert mixer.config.n_agents == 3
    assert mixer.config.state_dim == 75
    assert get_out_feat(mixer.hyper_w_1) == 3 * 32
    assert get_out_feat(mixer.hyper_w_final) == config.embed_dim
    assert get_out_feat(mixer.hyper_b_1) == config.embed_dim
    assert get_out_feat(mixer.V) == 1

    assert isinstance(mixer.V, nn.Sequential)
