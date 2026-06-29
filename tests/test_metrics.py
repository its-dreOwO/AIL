import numpy as np

from forecasting.metrics import calc_best_of_k_mpjpe, calc_mpjpe


def _entry(offset, target_pid, pids):
    T, P = 250, len(pids)
    gt = np.zeros((T, P, 29, 3), dtype=np.float32)
    pred = np.zeros((T, P, 29, 3), dtype=np.float32)
    idx = pids.index(target_pid)
    pred[:, idx] = gt[:, idx] + offset
    masks = np.ones((T, P), dtype=np.float32)
    return {
        "Poses3d_out": gt,
        "Masks_out": masks,
        "Poses3d_out_pred": pred,
        "target_pid": target_pid,
        "pids": pids,
    }


def test_overall_mean_matches_constant_error():
    off = np.array([0.3, 0.4, 0.0], dtype=np.float32)
    results = {"walking": [_entry(off, 7, [7, 9])]}
    out = calc_mpjpe(results)
    assert abs(out["overall"]["mean"] - 0.5) < 1e-5
    assert abs(out["overall"]["at_sec"][10] - 0.5) < 1e-5
    assert abs(out["per_action"]["walking"]["mean"] - 0.5) < 1e-5


def test_target_pid_selection():
    off = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e = _entry(np.zeros(3, dtype=np.float32), 7, [7, 9])
    e["Poses3d_out_pred"][:, 1] += off
    out = calc_mpjpe({"walking": [e]})
    assert out["overall"]["mean"] == 0.0


def test_best_of_k_selects_lowest_error_sample_per_entry():
    e = _entry(np.zeros(3, dtype=np.float32), 7, [7])
    gt = e["Poses3d_out"]
    bad = gt + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    good = gt + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    e["Poses3d_out_pred_samples"] = np.stack([bad, good], axis=0)

    out = calc_best_of_k_mpjpe({"walking": [e]})

    assert abs(out["overall"]["mean"] - 0.1) < 1e-5
    assert out["overall"]["best_sample_indices"] == [1]
    assert out["per_action"]["walking"]["best_sample_indices"] == [1]


def test_best_of_k_respects_target_pid_and_masks():
    e = _entry(np.zeros(3, dtype=np.float32), 9, [7, 9])
    gt = e["Poses3d_out"]
    bad = gt.copy()
    good = gt.copy()
    bad[:, 1] += np.array([2.0, 0.0, 0.0], dtype=np.float32)
    good[:, 1] += np.array([0.2, 0.0, 0.0], dtype=np.float32)
    good[:, 0] += np.array([10.0, 0.0, 0.0], dtype=np.float32)
    e["Masks_out"][:125, 1] = 0.0
    e["Poses3d_out_pred_samples"] = np.stack([bad, good], axis=0)

    out = calc_best_of_k_mpjpe({"walking": [e]})

    assert abs(out["overall"]["mean"] - 0.2) < 1e-5
    assert out["overall"]["best_sample_indices"] == [1]
