import numpy as np

from forecasting.eval_bok import build_bok_results, capture_inputs


class FakeEvaluator:
    def __init__(self, cases):
        self._cases = cases

    def execute3d(self, callback_fn, **kwargs):
        results = {}
        for c in self._cases:
            results.setdefault(c["action"], []).append({})
            callback_fn(c)
        return results


def _case(action, target_pid, pids):
    T, P = 250, len(pids)
    return {
        "Poses3d_in": np.zeros((250, P, 29, 3), dtype=np.float32),
        "Masks_in": np.ones((250, P), dtype=np.float32),
        "Poses3d_out": np.zeros((T, P, 29, 3), dtype=np.float32),
        "Masks_out": np.ones((T, P), dtype=np.float32),
        "n_out": 250,
        "pids": pids,
        "frames_in": list(range(250)),
        "action": action,
        "pid": target_pid,
    }


def test_capture_inputs_collects_all_cases():
    cases = [_case("walking", 7, [7, 9]), _case("sink", 3, [3])]
    captured = capture_inputs(FakeEvaluator(cases))
    assert len(captured) == 2
    assert captured[0]["target_pid"] == 7
    assert captured[1]["action"] == "sink"


def test_build_bok_results_assembles_samples_and_scores():
    cases = [_case("walking", 7, [7])]
    captured = capture_inputs(FakeEvaluator(cases))

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
