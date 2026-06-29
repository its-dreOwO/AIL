import argparse
import os

import torch

from hik.eval.evaluator import Evaluator
from forecasting import config
from forecasting.callbacks import (
    make_scene_simlpe_callback,
    make_simlpe_callback,
    zero_velocity_callback,
)
from forecasting.eval_bok import evaluate_humanmac_bok, make_humanmac_sample_case
from forecasting.humanmac import HumanMAC
from forecasting.metrics import calc_mpjpe
from forecasting.model import SceneConditionedSiMLPe, SiMLPe


def evaluate_callback(callback_fn, dataset, data_path, test_json_path) -> dict:
    evaluator = Evaluator(test_json_path, dataset, data_path)
    results = evaluator.execute3d(callback_fn)
    return calc_mpjpe(results)


def load_model(
    ckpt, n_blocks=4, output_mode="velocity", device="cpu", model_name="simlpe"
):
    model_kwargs = dict(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
        output_mode=output_mode,
    )
    if model_name == "scene-simlpe":
        model = SceneConditionedSiMLPe(**model_kwargs)
    elif model_name == "simlpe":
        model = SiMLPe(**model_kwargs)
    else:
        raise ValueError("model_name must be 'simlpe' or 'scene-simlpe'")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def load_humanmac(ckpt, device="cpu", n_coeff=125, d_model=512, n_layers=8,
                  n_heads=8, timesteps=1000, ddim_steps=50):
    model = HumanMAC(
        pose_dim=config.POSE_DIM, window=config.WINDOW, n_in=config.N_IN,
        n_coeff=n_coeff, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        timesteps=timesteps, ddim_steps=ddim_steps,
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
    parser.add_argument(
        "--model",
        choices=["zerovel", "simlpe", "scene-simlpe", "humanmac"],
        default="zerovel",
    )
    parser.add_argument("--ckpt", default=os.path.join(config.cache_dir(), "simlpe.pt"))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-coeff", type=int, default=125)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument(
        "--output-mode",
        choices=["position", "velocity"],
        default="velocity",
        help="must match the output mode used for the checkpoint",
    )
    args = parser.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "humanmac":
        from hik.eval.evaluator import Evaluator

        model = load_humanmac(
            args.ckpt, device=device, n_coeff=args.n_coeff, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads, timesteps=args.timesteps,
            ddim_steps=args.ddim_steps,
        )
        sample_case = make_humanmac_sample_case(
            model, k=args.k, ddim_steps=args.ddim_steps, device=device
        )
        evaluator = Evaluator(test_json, args.dataset, config.data_root())
        bok, single = evaluate_humanmac_bok(evaluator, sample_case, k=args.k)
        print(f"\n=== humanmac on {args.dataset}  (K={args.k}, ddim_steps={args.ddim_steps}) ===")
        _print_report(f"best-of-{args.k} [ORACLE]", bok)
        _print_report("single-sample (sample 0)", single)
        return

    if args.model == "zerovel":
        callback = zero_velocity_callback
    else:
        model = load_model(
            args.ckpt,
            n_blocks=args.n_blocks,
            output_mode=args.output_mode,
            device=device,
            model_name=args.model,
        )
        if args.model == "scene-simlpe":
            callback = make_scene_simlpe_callback(model, device=device)
        else:
            callback = make_simlpe_callback(model, device=device)

    metrics = evaluate_callback(callback, args.dataset, config.data_root(), test_json)
    _print_report(f"{args.model} on {args.dataset}", metrics)


if __name__ == "__main__":
    main()
