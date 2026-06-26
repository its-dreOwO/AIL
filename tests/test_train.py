import numpy as np

from forecasting import config
from forecasting.train import train


def test_train_smoke_reduces_loss():
    rng = np.random.default_rng(0)
    n_windows = 64
    base = rng.standard_normal((n_windows, 29, 3)).astype(np.float32)
    t = np.linspace(0, 1, config.WINDOW, dtype=np.float32).reshape(
        1, config.WINDOW, 1, 1
    )
    vel = rng.standard_normal((n_windows, 29, 3)).astype(np.float32) * 0.1
    windows = (base[:, None] + (vel[:, None] * t)).astype(np.float32)

    out = train(
        windows, epochs=3, batch_size=16, lr=3e-3, device="cpu", n_blocks=2, out_path=None
    )
    assert len(out["train_loss"]) == 3
    assert out["train_loss"][-1] < out["train_loss"][0]
