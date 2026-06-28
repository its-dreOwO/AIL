import numpy as np
import torch

from forecasting import config
from forecasting.scene_features import SCENE_FEATURE_DIM
from forecasting.train import train, train_one_epoch


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


def test_train_uses_requested_output_mode(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, **kwargs):
            seen["output_mode"] = kwargs["output_mode"]

        def to(self, device):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr("forecasting.train.SiMLPe", FakeModel)
    monkeypatch.setattr("forecasting.train.train_one_epoch", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("forecasting.train.eval_loss", lambda *args, **kwargs: 1.0)

    class FakeOptim:
        def __init__(self, params, lr):
            pass

    class FakeSched:
        def __init__(self, optim, T_max):
            pass

        def step(self):
            pass

    monkeypatch.setattr("torch.optim.Adam", FakeOptim)
    monkeypatch.setattr("torch.optim.lr_scheduler.CosineAnnealingLR", FakeSched)

    windows = np.zeros((4, config.WINDOW, config.N_JOINTS, 3), dtype=np.float32)
    train(
        windows,
        epochs=1,
        batch_size=2,
        device="cpu",
        output_mode="velocity",
        out_path=None,
    )

    assert seen["output_mode"] == "velocity"


def test_train_one_epoch_passes_scene_batch_to_model():
    class SceneModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.seen_scene_shape = None

        def forward(self, x, scene):
            self.seen_scene_shape = tuple(scene.shape)
            return x * self.scale

    x = torch.randn(2, config.N_IN, config.POSE_DIM)
    scene = torch.randn(2, SCENE_FEATURE_DIM)
    y = torch.randn(2, config.N_OUT, config.POSE_DIM)
    loader = [(x, scene, y)]
    model = SceneModel()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_one_epoch(model, loader, optim, "cpu", vel_weight=0.0)

    assert model.seen_scene_shape == (2, SCENE_FEATURE_DIM)


def test_train_uses_scene_model_and_dataset(monkeypatch):
    seen = {}

    class FakeSceneModel:
        def __init__(self, **kwargs):
            seen["model"] = kwargs["output_mode"]

        def to(self, device):
            return self

        def parameters(self):
            return []

    class FakeDataset:
        def __init__(self, windows, scene):
            seen["dataset_shapes"] = (windows.shape, scene.shape)

        def __len__(self):
            return 4

    monkeypatch.setattr("forecasting.train.SceneConditionedSiMLPe", FakeSceneModel)
    monkeypatch.setattr("forecasting.train.SceneWindowDataset", FakeDataset)
    monkeypatch.setattr("forecasting.train.train_one_epoch", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("forecasting.train.eval_loss", lambda *args, **kwargs: 1.0)

    class FakeOptim:
        def __init__(self, params, lr):
            pass

    class FakeSched:
        def __init__(self, optim, T_max):
            pass

        def step(self):
            pass

    monkeypatch.setattr("torch.optim.Adam", FakeOptim)
    monkeypatch.setattr("torch.optim.lr_scheduler.CosineAnnealingLR", FakeSched)

    windows = np.zeros((4, config.WINDOW, config.N_JOINTS, 3), dtype=np.float32)
    scene = np.zeros((4, SCENE_FEATURE_DIM), dtype=np.float32)
    train(
        (windows, scene),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_name="scene-simlpe",
        output_mode="position",
        out_path=None,
    )

    assert seen["model"] == "position"
    assert seen["dataset_shapes"] == (windows.shape, scene.shape)


from forecasting.train import train_humanmac


def test_train_humanmac_runs_one_epoch_and_writes_ckpt(tmp_path):
    windows = np.random.default_rng(0).standard_normal(
        (6, config.WINDOW, 29, 3)
    ).astype(np.float32)
    out = tmp_path / "humanmac.pt"
    hist = train_humanmac(
        windows, epochs=1, batch_size=3, device="cpu",
        n_coeff=8, d_model=16, n_layers=1, n_heads=2,
        timesteps=20, ddim_steps=2, out_path=str(out),
    )
    assert len(hist["train_loss"]) == 1
    assert np.isfinite(hist["train_loss"][0])
    assert out.exists()
