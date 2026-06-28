import numpy as np

from forecasting.callbacks import zero_velocity_callback
from forecasting.metrics import calc_best_of_k_mpjpe, calc_mpjpe


def capture_inputs(evaluator):
    cases = []

    def callback(inp):
        case = dict(inp)
        case["target_pid"] = inp["pid"]
        cases.append(case)
        return zero_velocity_callback(inp)

    evaluator.execute3d(callback)
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
