import argparse
import os

import torch

from hik.eval.evaluator import Evaluator
from forecasting import config
from forecasting.callbacks import make_simlpe_callback, zero_velocity_callback
from forecasting.metrics import calc_mpjpe
from forecasting.model import SiMLPe


def evaluate_callback(callback_fn, dataset, data_path, test_json_path) -> dict:
    evaluator = Evaluator(test_json_path, dataset, data_path)
    results = evaluator.execute3d(callback_fn)
    return calc_mpjpe(results)


def load_model(ckpt, n_blocks=4, device="cpu") -> SiMLPe:
    model = SiMLPe(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def _print_report(name, metrics):
    overall = metrics["overall"]
    print(f"\n=== {name} ===")
    print(f"overall mean MPJPE: {overall['mean']:.4f}")
    for seconds in (1, 5, 10):
        print(f"  @{seconds}s: {overall['at_sec'][seconds]:.4f}")
    for action, values in metrics["per_action"].items():
        print(f"  [{action}] mean {values['mean']:.4f}  @10s {values['at_sec'][10]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="A")
    parser.add_argument("--model", choices=["zerovel", "simlpe"], default="zerovel")
    parser.add_argument("--ckpt", default=os.path.join(config.cache_dir(), "simlpe.pt"))
    parser.add_argument("--n-blocks", type=int, default=4)
    args = parser.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "zerovel":
        callback = zero_velocity_callback
    else:
        model = load_model(args.ckpt, n_blocks=args.n_blocks, device=device)
        callback = make_simlpe_callback(model, device=device)

    metrics = evaluate_callback(callback, args.dataset, config.data_root(), test_json)
    _print_report(f"{args.model} on {args.dataset}", metrics)


if __name__ == "__main__":
    main()
