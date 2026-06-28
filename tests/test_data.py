import os

import numpy as np
import pytest
import torch

from forecasting import config
from forecasting.data import (
    SceneWindowDataset,
    WindowDataset,
    build_or_load_scene_windows,
    build_windows_for_dataset,
    forbidden_frames,
    scene_windows_cache_path,
    window_is_clean,
)
from forecasting.scene_features import SCENE_FEATURE_DIM

TEST_JSON = os.path.join(config._REPO_ROOT, "testdata", "test.json")


def test_forbidden_frames_covers_range():
    fb = forbidden_frames("A", TEST_JSON)
    assert len(fb) > 0
    assert isinstance(fb, set)


def test_window_is_clean_rejects_overlap_and_gaps():
    full = np.ones(config.WINDOW, dtype=np.float32)
    forbidden = {1000}
    assert window_is_clean(0, full, forbidden) is True
    assert window_is_clean(900, full, forbidden) is False
    gappy = full.copy()
    gappy[10] = 0.0
    assert window_is_clean(0, gappy, set()) is False


def test_window_dataset_splits_in_out():
    arr = np.random.default_rng(0).standard_normal(
        (3, config.WINDOW, 29, 3)
    ).astype(np.float32)
    ds = WindowDataset(arr)
    x, y = ds[0]
    assert x.shape == (config.N_IN, config.POSE_DIM)
    assert y.shape == (config.N_OUT, config.POSE_DIM)
    assert torch.is_tensor(x) and x.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not config.dataset_available(), reason="dataset not mounted")
def test_build_windows_real_small():
    w = build_windows_for_dataset(
        "A", stepsize=500, test_json_path=TEST_JSON, max_windows=8
    )
    assert w.ndim == 4 and w.shape[1:] == (config.WINDOW, 29, 3)
    assert 0 < w.shape[0] <= 8
    assert np.isfinite(w).all()


def test_scene_window_dataset_splits_pose_and_scene_features():
    windows = np.random.default_rng(1).standard_normal(
        (3, config.WINDOW, 29, 3)
    ).astype(np.float32)
    scene = np.random.default_rng(2).standard_normal(
        (3, SCENE_FEATURE_DIM)
    ).astype(np.float32)

    ds = SceneWindowDataset(windows, scene)
    x, s, y = ds[1]

    assert x.shape == (config.N_IN, config.POSE_DIM)
    assert s.shape == (SCENE_FEATURE_DIM,)
    assert y.shape == (config.N_OUT, config.POSE_DIM)
    assert torch.is_tensor(x) and x.dtype == torch.float32
    assert torch.is_tensor(s) and s.dtype == torch.float32
    assert torch.is_tensor(y) and y.dtype == torch.float32


def test_scene_window_dataset_rejects_mismatched_lengths():
    windows = np.zeros((2, config.WINDOW, 29, 3), dtype=np.float32)
    scene = np.zeros((3, SCENE_FEATURE_DIM), dtype=np.float32)

    try:
        SceneWindowDataset(windows, scene)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scene_windows_cache_path_is_distinct_from_pose_window_cache():
    path = scene_windows_cache_path(["A", "B"], stepsize=50)

    assert path.endswith("scene_windows_AB_s50_v1.npz")
    assert "windows_AB_s50.npy" not in path


def test_build_or_load_scene_windows_loads_npz_cache(monkeypatch, tmp_path):
    cached_windows = np.zeros((2, config.WINDOW, 29, 3), dtype=np.float32)
    cached_scene = np.ones((2, SCENE_FEATURE_DIM), dtype=np.float32)
    cache_path = tmp_path / "scene_windows_AB_s50_v1.npz"
    np.savez_compressed(cache_path, windows=cached_windows, scene=cached_scene)

    monkeypatch.setattr(
        "forecasting.data.scene_windows_cache_path",
        lambda datasets, stepsize: str(cache_path),
    )

    windows, scene = build_or_load_scene_windows(["A", "B"], 50, TEST_JSON)

    assert np.array_equal(windows, cached_windows)
    assert np.array_equal(scene, cached_scene)


def test_build_or_load_scene_windows_saves_npz_cache(monkeypatch, tmp_path):
    built_windows = np.zeros((1, config.WINDOW, 29, 3), dtype=np.float32)
    built_scene = np.ones((1, SCENE_FEATURE_DIM), dtype=np.float32)
    cache_path = tmp_path / "scene_windows_A_s50_v1.npz"

    monkeypatch.setattr(
        "forecasting.data.scene_windows_cache_path",
        lambda datasets, stepsize: str(cache_path),
    )
    monkeypatch.setattr(
        "forecasting.data.build_scene_windows_for_dataset",
        lambda dataset, stepsize, test_json_path: (built_windows, built_scene),
    )

    windows, scene = build_or_load_scene_windows(["A"], 50, TEST_JSON)

    assert np.array_equal(windows, built_windows)
    assert np.array_equal(scene, built_scene)
    loaded = np.load(cache_path)
    assert np.array_equal(loaded["windows"], built_windows)
    assert np.array_equal(loaded["scene"], built_scene)


from forecasting.data import WholeWindowDataset


def test_whole_window_dataset_returns_full_window_x0():
    windows = np.random.default_rng(3).standard_normal(
        (4, config.WINDOW, 29, 3)
    ).astype(np.float32)
    ds = WholeWindowDataset(windows)
    assert len(ds) == 4
    x0 = ds[2]
    assert x0.shape == (config.WINDOW, config.POSE_DIM)
    assert torch.is_tensor(x0) and x0.dtype == torch.float32
