import numpy as np
import torch

from forecasting.callbacks import (
    make_scene_simlpe_callback,
    make_simlpe_callback,
    zero_velocity_callback,
)
from forecasting.scene_features import SCENE_FEATURE_DIM


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


class RecordingSceneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_shape = None

    def forward(self, x, scene):
        self.scene_shape = tuple(scene.shape)
        return x


def test_scene_simlpe_callback_shape_and_scene_tensor():
    class FakeObject:
        isbox = False
        location = np.array([2.0, 0.0, 0.0, 0.5], dtype=np.float32)
        label = np.eye(13, dtype=np.float32)[10]

    class FakeKitchen:
        def get_environment(self, frame, ignore_oob=True, use_pointcloud=False):
            return [FakeObject()]

    model = RecordingSceneModel()
    cb = make_scene_simlpe_callback(model, device="cpu")
    inp = _make_input(P=1)
    inp["kitchen"] = FakeKitchen()

    out = cb(inp)

    assert out.shape == (250, 1, 29, 3)
    assert np.isfinite(out).all()
    assert model.scene_shape == (1, SCENE_FEATURE_DIM)
