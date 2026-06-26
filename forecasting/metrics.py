import numpy as np

from hik.eval.mpjpe import mean_per_joint_l2_distance
from forecasting.config import FPS


def _entry_curve(entry):
    pids = list(entry["pids"])
    idx = pids.index(entry["target_pid"])
    gt = entry["Poses3d_out"][:, idx]
    pred = entry["Poses3d_out_pred"][:, idx]
    mask = entry["Masks_out"][:, idx]
    curve = mean_per_joint_l2_distance(gt, pred)
    return np.where(mask > 0.5, curve, np.nan)


def _summarize(curves, horizons_sec):
    stack = np.stack(curves, axis=0)
    curve = np.nanmean(stack, axis=0)
    at_sec = {}
    for s in horizons_sec:
        frame = min(s * FPS - 1, len(curve) - 1)
        at_sec[s] = float(curve[frame])
    return {
        "curve": [float(x) for x in curve],
        "at_sec": at_sec,
        "mean": float(np.nanmean(curve)),
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
