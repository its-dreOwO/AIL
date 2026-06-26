import os

import numpy as np
import pytest
import torch

from forecasting import config
from forecasting.data import (
    WindowDataset,
    build_windows_for_dataset,
    forbidden_frames,
    window_is_clean,
)

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
