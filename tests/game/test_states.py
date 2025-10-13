import numpy as np
import pytest

from src.config.qmix_base import QmixBaseConfig
from src.game.experiment import Experiment


def create_config():
    return QmixBaseConfig(
        embed_dim=32,
        input_dim=4,
        hidden_dims=[64, 32],
        hypernet_embed=64,
        n_actions=3,
        state_shape=(4,),
    )


def test_normalize_feature_w_ints():
    experiment = Experiment(_config=create_config())
    assert experiment._normalize_feature(10, 5, 15) == 0
    assert experiment._normalize_feature(5, 5, 15) == -1
    assert experiment._normalize_feature(15, 5, 15) == 1
    assert experiment._normalize_feature(8, 5, 15) == -0.4
    assert experiment._normalize_feature(12, 5, 15) == pytest.approx(0.4)


def test_normalize_feature_w_floats():
    experiment = Experiment(_config=create_config())
    assert experiment._normalize_feature(10.5, 5, 15) == pytest.approx(0.1)
    assert experiment._normalize_feature(5.5, 5, 15) == pytest.approx(-0.9)
    assert experiment._normalize_feature(13.5, 5, 15) == pytest.approx(0.7)
    assert experiment._normalize_feature(8.3, 5, 15) == pytest.approx(-0.34)
    assert experiment._normalize_feature(12.3, 5, 15) == pytest.approx(0.46)


def test_normalize_feature_out_of_bounds():
    experiment = Experiment(_config=create_config())
    assert experiment._normalize_feature(3, 5, 15) == -1
    assert experiment._normalize_feature(3.5, 5, 15) == -1
    assert experiment._normalize_feature(20, 5, 15) == 1
    assert experiment._normalize_feature(20.5, 5, 15) == 1


def test_normalize_state_shape_and_type():
    experiment = Experiment(_config=create_config())
    obs = [0.0] * 8
    normalized = experiment._normalize_global_state(obs)

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == (8,)


def test_normalize_state_in_bounds():
    experiment = Experiment(_config=create_config())
    obs = [0.0] * 8
    normalized = experiment._normalize_global_state(obs)

    assert np.allclose(normalized, 0.0, atol=1e-6)


def test_normalize_state_at_edges():
    experiment = Experiment(_config=create_config())
    obs = [-2, 2, -4, 4, -30, 30, -1.9, 1.9]
    normalized = experiment._normalize_global_state(obs)

    expected = [-1, 1, -1, 1, -1, 1, -1, 1]
    assert np.allclose(normalized, expected)


def test_normalize_state_clipping():
    experiment = Experiment(_config=create_config())
    obs = [-10, 10, -10, 10, -90, 90, -10, 10]
    normalized = experiment._normalize_global_state(obs)

    assert np.all(normalized <= 1.3)
    assert np.all(normalized >= -1.3)


def test_normalize_state_mixed():
    experiment = Experiment(_config=create_config())
    obs = [1.0, -1.0, 2.0, -2.0, 15.0, -15.0, 1.0, -1.0]
    normalized = experiment._normalize_global_state(obs)

    assert len(normalized) == 8

    # 1.0 in [-2,2]
    assert normalized[0] == pytest.approx(0.5)

    # 15.0 in [-30,30]
    assert normalized[4] == pytest.approx(0.5)


def test_global_observation_property_sets_and_normalizes():
    obs = [-2, 2, -4, 4, -30, 30, -1.9, 1.9]
    experiment = Experiment(_config=create_config())
    experiment.global_observation = (obs, [])

    assert np.allclose(
        experiment.global_observation, [-1, 1, -1, 1, -1, 1, -1, 1]
    )
