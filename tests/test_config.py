import os

from forecasting import config


def test_constants():
    assert config.N_IN == 250
    assert config.N_OUT == 250
    assert config.WINDOW == 500
    assert config.POSE_DIM == config.N_JOINTS * 3 == 87
    assert config.DATASETS == ("A", "B", "C", "D")


def test_data_root_from_env(monkeypatch):
    monkeypatch.setenv("HIK_DATA", "/tmp/somewhere")
    assert config.data_root() == "/tmp/somewhere"
    assert config.poses_path() == "/tmp/somewhere/poses"
    assert config.smplx_path().endswith("body_models/SMPLX_NEUTRAL.npz")


def test_cache_dir_created():
    d = config.cache_dir()
    assert os.path.isdir(d)
