import warnings

import numpy as np

from hik.eval.mpjpe import mean_per_joint_l2_distance
from forecasting.config import FPS


def _nanmean(a, axis=None):
    # Fully-masked frames are all-NaN; NaN is the intended result, not a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _curve(entry, pred_block):
    pids = list(entry["pids"])
    idx = pids.index(entry["target_pid"])
    gt = entry["Poses3d_out"][:, idx]
    pred = pred_block[:, idx]
    mask = entry["Masks_out"][:, idx]
    curve = mean_per_joint_l2_distance(gt, pred)
    return np.where(mask > 0.5, curve, np.nan)


def _entry_curve(entry):
    return _curve(entry, entry["Poses3d_out_pred"])


def _best_of_k_curve(entry):
    samples = np.asarray(entry["Poses3d_out_pred_samples"])
    curves = [_curve(entry, samples[k]) for k in range(len(samples))]
    best = int(np.nanargmin([_nanmean(c) for c in curves]))
    return curves[best], best


def _summarize(curves, horizons_sec):
    stack = np.stack(curves, axis=0)
    curve = _nanmean(stack, axis=0)
    at_sec = {}
    for s in horizons_sec:
        frame = min(s * FPS - 1, len(curve) - 1)
        at_sec[s] = float(curve[frame])
    return {
        "curve": [float(x) for x in curve],
        "at_sec": at_sec,
        "mean": float(_nanmean(curve)),
    }


def calc_mpjpe(results: dict, horizons_sec=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) -> dict:
    per_action = {}
    all_curves = []
    for action, entries in results.items():
        curves = [_entry_curve(entry) for entry in entries]
        all_curves.extend(curves)
        per_action[action] = _summarize(curves, horizons_sec)
    overall = _summarize(all_curves, horizons_sec)
    return {"per_action": per_action, "overall": overall}


def calc_best_of_k_mpjpe(
    results: dict, horizons_sec=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
) -> dict:
    """Best-of-K MPJPE: per entry, score the lowest-error sample from
    ``Poses3d_out_pred_samples`` (shape ``(K, T, P, 29, 3)``)."""
    per_action = {}
    all_curves = []
    all_indices = []
    for action, entries in results.items():
        curves = []
        indices = []
        for entry in entries:
            curve, best = _best_of_k_curve(entry)
            curves.append(curve)
            indices.append(best)
        all_curves.extend(curves)
        all_indices.extend(indices)
        summary = _summarize(curves, horizons_sec)
        summary["best_sample_indices"] = indices
        per_action[action] = summary
    overall = _summarize(all_curves, horizons_sec)
    overall["best_sample_indices"] = all_indices
    return {"per_action": per_action, "overall": overall}
