import numpy as np
import torch

from forecasting.callbacks import make_simlpe_callback, zero_velocity_callback


def _make_input(P=2):
    rng = np.random.default_rng(1)
    poses = rng.standard_normal((250, P, 29, 3)).astype(np.float32) + 1.0
    poses[:, :, :, 2] += 1.0
    masks = np.ones((250, P), dtype=np.float32)
    return {
        "Poses3d_in": poses,
        "Masks_in": masks,
        "n_out": 250,
        "pids": list(range(P)),
        "kitchen": None,
        "frames_in": list(range(250)),
        "action": "walking",
    }


def test_zero_velocity_shape_and_values():
    inp = _make_input()
    out = zero_velocity_callback(inp)
    assert out.shape == (250, 2, 29, 3)
    assert np.allclose(out, inp["Poses3d_in"][-1][None], atol=1e-6)


def test_simlpe_callback_shape_and_inversion():
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    cb = make_simlpe_callback(IdentityModel(), device="cpu")
    inp = _make_input()
    out = cb(inp)
    assert out.shape == (250, 2, 29, 3)
    assert np.isfinite(out).all()
