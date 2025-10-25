import torch

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.qmix.agent_networks import (
    QmixGRUNetwork,
    QmixQNetNetwork,
)


def create_config():
    return QmixBaseConfig(
        embed_dim=32,
        input_dim=4,
        hidden_dims=[64, 32],
        hypernet_embed=64,
        n_actions=3,
        state_shape=(4,),
    )


def test_qmix_qnet_network_output_shape():
    config = create_config()
    model = QmixQNetNetwork(config)
    obs = torch.rand(1, config.input_dim)
    out, _ = model(obs)
    assert out.shape == (1, config.n_actions)


def skip_test_qmix_gru_network_output_shape():
    config = create_config()
    model = QmixGRUNetwork(config)
    obs = torch.rand(1, config.input_dim)
    out, _ = model(obs)
    assert out.shape == (1, config.n_actions)


def test_qmix_qnet_device_consistency():
    config = create_config()
    model = QmixQNetNetwork(config)
    obs = torch.rand(1, config.input_dim)
    out, _ = model(obs)
    assert out.device.type == torch.device(config.device).type


def skip_test_qmix_gru_device_consistency():
    config = create_config()
    model = QmixGRUNetwork(config)
    obs = torch.rand(1, config.input_dim)
    out, _ = model(obs)
    assert out.device.type == torch.device(config.device).type
