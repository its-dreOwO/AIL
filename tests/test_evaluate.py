import numpy as np

from forecasting import evaluate
from forecasting.callbacks import zero_velocity_callback


def test_evaluate_callback_uses_metrics(monkeypatch):
    fake_results = {
        "walking": [
            {
                "Poses3d_out": np.zeros((250, 1, 29, 3), np.float32),
                "Masks_out": np.ones((250, 1), np.float32),
                "Poses3d_out_pred": np.zeros((250, 1, 29, 3), np.float32),
                "target_pid": 5,
                "pids": [5],
            }
        ]
    }

    class FakeEval:
        def __init__(self, *args, **kwargs):
            pass

        def execute3d(self, callback_fn, **kwargs):
            return fake_results

    monkeypatch.setattr(evaluate, "Evaluator", FakeEval)
    out = evaluate.evaluate_callback(zero_velocity_callback, "A", "/nope", "/nope")
    assert out["overall"]["mean"] == 0.0
    assert 10 in out["overall"]["at_sec"]


def test_load_model_uses_requested_output_mode(monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, **kwargs):
            seen["output_mode"] = kwargs["output_mode"]

        def load_state_dict(self, state):
            seen["state"] = state

    monkeypatch.setattr(evaluate, "SiMLPe", FakeModel)
    monkeypatch.setattr(evaluate.torch, "load", lambda *args, **kwargs: {"ok": True})

    model = evaluate.load_model("checkpoint.pt", output_mode="position")

    assert model is not None
    assert seen == {"output_mode": "position", "state": {"ok": True}}


def test_load_model_uses_scene_model(monkeypatch):
    seen = {}

    class FakeSceneModel:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def load_state_dict(self, state):
            seen["state"] = state

    monkeypatch.setattr(evaluate, "SceneConditionedSiMLPe", FakeSceneModel)
    monkeypatch.setattr(evaluate.torch, "load", lambda *args, **kwargs: {"ok": True})

    model = evaluate.load_model(
        "checkpoint.pt", model_name="scene-simlpe", output_mode="position"
    )

    assert model is not None
    assert seen["kwargs"]["output_mode"] == "position"
    assert seen["state"] == {"ok": True}
