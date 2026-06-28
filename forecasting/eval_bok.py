import numpy as np
import torch

from hik.transforms.utils import backfill_masked, denormalize3d, normalize3d
from forecasting.callbacks import zero_velocity_callback
from forecasting.config import N_IN, POSE_DIM
from forecasting.metrics import calc_best_of_k_mpjpe, calc_mpjpe


def capture_inputs(evaluator):
    # The vendored Evaluator does not expose target_pid to the callback, but its
    # returned results entries do (plus Poses3d_in/Masks_in). Run one cheap
    # zero-velocity enumeration pass and read the entries directly.
    results = evaluator.execute3d(zero_velocity_callback)
    cases = []
    for action, entries in results.items():
        for entry in entries:
            case = dict(entry)
            case["action"] = action
            case["n_out"] = entry["Poses3d_out"].shape[0]
            cases.append(case)
    return cases


def build_bok_results(cases, sample_case, k):
    results_bok = {}
    results_single = {}
    for inp in cases:
        action = inp["action"]
        samples = np.asarray(sample_case(inp), dtype=np.float32)  # [k, n_out, P, 29, 3]
        base = {
            "Poses3d_in": inp["Poses3d_in"],
            "Masks_in": inp["Masks_in"],
            "Poses3d_out": inp["Poses3d_out"],
            "Masks_out": inp["Masks_out"],
            "frames_in": inp.get("frames_in"),
            "target_pid": inp["target_pid"],
            "pids": inp["pids"],
        }
        bok_entry = dict(base)
        bok_entry["Poses3d_out_pred_samples"] = samples
        single_entry = dict(base)
        single_entry["Poses3d_out_pred"] = samples[0]
        results_bok.setdefault(action, []).append(bok_entry)
        results_single.setdefault(action, []).append(single_entry)
    return results_bok, results_single


def evaluate_humanmac_bok(evaluator, sample_case, k):
    cases = capture_inputs(evaluator)
    results_bok, results_single = build_bok_results(cases, sample_case, k)
    return calc_best_of_k_mpjpe(results_bok), calc_mpjpe(results_single)


def make_humanmac_sample_case(model, k, ddim_steps=None, device="cpu"):
    model = model.to(device)
    model.eval()

    def sample_case(inp):
        poses_in = np.copy(inp["Poses3d_in"])
        masks_in = np.copy(inp["Masks_in"])
        n_out = inp["n_out"]
        pids = inp["pids"]
        zv = zero_velocity_callback(inp)  # [n_out, P, 29, 3]
        out = np.repeat(zv[None], k, axis=0).astype(np.float32)  # [k, n_out, P, 29, 3]

        target_idx = pids.index(inp["target_pid"])
        filled, _ = backfill_masked(poses_in, masks_in)
        block = filled[:, target_idx : target_idx + 1]
        try:
            normed, norm_params = normalize3d(block, frame=N_IN - 1)
        except Exception:
            return out  # fall back to zero-velocity for all samples

        obs = torch.tensor(
            normed[:N_IN, 0].reshape(N_IN, POSE_DIM), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            fut = model.sample(obs, k=k, ddim_steps=ddim_steps)  # [k, n_out, POSE_DIM]
        fut = fut.cpu().numpy().reshape(k, n_out, 1, 29, 3)
        for s in range(k):
            out[s, :, target_idx : target_idx + 1] = denormalize3d(fut[s], norm_params)
        return out

    return sample_case
