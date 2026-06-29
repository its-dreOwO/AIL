import numpy as np
import torch

from forecasting.eval_bok import (
    build_bok_results,
    capture_inputs,
    make_humanmac_sample_case,
)
from forecasting.humanmac import HumanMAC


class FakeEvaluator:
    """Mimics hik Evaluator: execute3d returns {action: [entry, ...]} where each
    entry carries target_pid/pids/Poses3d_in/Masks_in/Poses3d_out/Masks_out."""

    def __init__(self, results):
        self._results = results

    def execute3d(self, callback_fn, **kwargs):
        return self._results


def _entry(target_pid, pids):
    T, P = 250, len(pids)
    return {
        "Poses3d_in": np.zeros((250, P, 29, 3), dtype=np.float32),
        "Masks_in": np.ones((250, P), dtype=np.float32),
        "Poses3d_out": np.zeros((T, P, 29, 3), dtype=np.float32),
        "Masks_out": np.ones((T, P), dtype=np.float32),
        "frames_in": list(range(250)),
        "Poses3d_out_pred": np.zeros((T, P, 29, 3), dtype=np.float32),
        "target_pid": target_pid,
        "pids": pids,
    }


def test_capture_inputs_collects_all_cases():
    results = {"walking": [_entry(7, [7, 9])], "sink": [_entry(3, [3])]}
    captured = capture_inputs(FakeEvaluator(results))
    assert len(captured) == 2
    assert captured[0]["target_pid"] == 7
    assert captured[0]["action"] == "walking"
    assert captured[0]["n_out"] == 250
    assert captured[1]["action"] == "sink"


def test_build_bok_results_assembles_samples_and_scores():
    results = {"walking": [_entry(7, [7])]}
    captured = capture_inputs(FakeEvaluator(results))

    def sample_case(inp):
        k, n_out, P = 2, inp["n_out"], len(inp["pids"])
        out = np.zeros((k, n_out, P, 29, 3), dtype=np.float32)
        out[0] += 1.0  # bad sample
        out[1] += 0.1  # good sample
        return out

    results_bok, results_single = build_bok_results(captured, sample_case, k=2)
    entry = results_bok["walking"][0]
    assert entry["Poses3d_out_pred_samples"].shape == (2, 250, 1, 29, 3)
    assert results_single["walking"][0]["Poses3d_out_pred"].shape == (250, 1, 29, 3)


def test_humanmac_sample_case_shapes_and_target_only():
    model = HumanMAC(
        pose_dim=87, window=500, n_in=250, n_coeff=8,
        d_model=16, n_layers=1, n_heads=2, timesteps=20, ddim_steps=2,
    )
    sample_case = make_humanmac_sample_case(model, k=2, device="cpu")
    inp = _entry(7, [7, 9])
    inp["n_out"] = 250
    out = sample_case(inp)
    assert out.shape == (2, 250, 2, 29, 3)
    assert np.isfinite(out).all()
    # non-target person (pid 9, index 1) is identical across samples (zero-velocity)
    assert np.allclose(out[0, :, 1], out[1, :, 1])
