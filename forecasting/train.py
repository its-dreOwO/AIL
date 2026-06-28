import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from forecasting import config
from forecasting.data import (
    SceneWindowDataset,
    WholeWindowDataset,
    WindowDataset,
    build_or_load_scene_windows,
    build_or_load_windows,
)
from forecasting.losses import horizon_weights, mpjpe_loss, velocity_loss
from forecasting.model import SceneConditionedSiMLPe, SiMLPe
from forecasting.humanmac import HumanMAC


def _split_batch(batch, device):
    if len(batch) == 2:
        x, y = batch
        return x.to(device), None, y.to(device)
    x, scene, y = batch
    return x.to(device), scene.to(device), y.to(device)


def _predict(model, x, scene):
    if scene is None:
        return model(x)
    return model(x, scene)


def train_one_epoch(
    model, loader, optim, device, vel_weight=1.0, pos_w=None, vel_w=None
):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x, scene, y = _split_batch(batch, device)
        optim.zero_grad()
        pred = _predict(model, x, scene)
        loss = mpjpe_loss(pred, y, weights=pos_w) + vel_weight * velocity_loss(
            pred, y, weights=vel_w
        )
        loss.backward()
        optim.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, vel_weight=1.0, pos_w=None, vel_w=None):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x, scene, y = _split_batch(batch, device)
        pred = _predict(model, x, scene)
        loss = mpjpe_loss(pred, y, weights=pos_w) + vel_weight * velocity_loss(
            pred, y, weights=vel_w
        )
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def train(
    windows,
    *,
    epochs,
    batch_size=256,
    lr=5e-4,
    val_frac=0.05,
    vel_weight=0.2,
    horizon_floor=0.2,
    device=None,
    n_blocks=4,
    output_mode="velocity",
    model_name="simlpe",
    seed=0,
    out_path="__default__",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if out_path == "__default__":
        ckpt_name = "scene_simlpe.pt" if model_name == "scene-simlpe" else "simlpe.pt"
        out_path = os.path.join(config.cache_dir(), ckpt_name)

    torch.manual_seed(seed)
    if model_name == "scene-simlpe":
        pose_windows, scene_features = windows
        dataset = SceneWindowDataset(pose_windows, scene_features)
        model = SceneConditionedSiMLPe(
            t_in=config.N_IN,
            t_out=config.N_OUT,
            pose_dim=config.POSE_DIM,
            n_blocks=n_blocks,
            output_mode=output_mode,
        ).to(device)
    elif model_name == "simlpe":
        dataset = WindowDataset(windows)
        model = SiMLPe(
            t_in=config.N_IN,
            t_out=config.N_OUT,
            pose_dim=config.POSE_DIM,
            n_blocks=n_blocks,
            output_mode=output_mode,
        ).to(device)
    else:
        raise ValueError("model_name must be 'simlpe' or 'scene-simlpe'")
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    pos_w = horizon_weights(config.N_OUT, horizon_floor, device=device)
    vel_w = horizon_weights(config.N_OUT - 1, horizon_floor, device=device)

    hist = {"train_loss": [], "val_loss": [], "best_val": float("inf"), "ckpt": None}
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            vel_weight=vel_weight,
            pos_w=pos_w,
            vel_w=vel_w,
        )
        val_loss = eval_loss(
            model,
            val_loader,
            device,
            vel_weight=vel_weight,
            pos_w=pos_w,
            vel_w=vel_w,
        )
        sched.step()
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        if val_loss < hist["best_val"]:
            hist["best_val"] = val_loss
            if out_path is not None:
                torch.save(model.state_dict(), out_path)
                hist["ckpt"] = out_path
        print(f"epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")
    return hist


def train_humanmac(
    windows,
    *,
    epochs,
    batch_size=64,
    lr=2e-4,
    val_frac=0.05,
    device=None,
    n_coeff=125,
    d_model=512,
    n_layers=8,
    n_heads=8,
    timesteps=1000,
    ddim_steps=50,
    seed=0,
    out_path="__default__",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if out_path == "__default__":
        out_path = os.path.join(config.cache_dir(), "humanmac.pt")

    torch.manual_seed(seed)
    dataset = WholeWindowDataset(windows)
    model = HumanMAC(
        pose_dim=config.POSE_DIM, window=config.WINDOW, n_in=config.N_IN,
        n_coeff=n_coeff, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        timesteps=timesteps, ddim_steps=ddim_steps,
    ).to(device)

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    hist = {"train_loss": [], "val_loss": [], "best_val": float("inf"), "ckpt": None}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x0 in train_loader:
            x0 = x0.to(device)
            optim.zero_grad()
            loss = model(x0)
            loss.backward()
            optim.step()
            total += loss.item() * x0.shape[0]
            n += x0.shape[0]
        train_loss = total / max(n, 1)

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for x0 in val_loader:
                x0 = x0.to(device)
                vtotal += model(x0).item() * x0.shape[0]
                vn += x0.shape[0]
        val_loss = vtotal / max(vn, 1)

        sched.step()
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        if val_loss < hist["best_val"]:
            hist["best_val"] = val_loss
            if out_path is not None:
                torch.save(model.state_dict(), out_path)
                hist["ckpt"] = out_path
        print(f"epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["simlpe", "scene-simlpe", "humanmac"], default="simlpe"
    )
    parser.add_argument("--n-coeff", type=int, default=125)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    parser.add_argument("--stepsize", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vel-weight", type=float, default=0.2)
    parser.add_argument("--horizon-floor", type=float, default=0.2)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument(
        "--output-mode",
        choices=["position", "velocity"],
        default="velocity",
        help="decode raw model output as last-frame residual positions or integrated velocities",
    )
    args = parser.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    if args.model == "humanmac":
        windows = build_or_load_windows(args.datasets, args.stepsize, test_json)
        print(f"training humanmac on {len(windows)} windows "
              f"(n_coeff={args.n_coeff}, timesteps={args.timesteps}, "
              f"ddim_steps={args.ddim_steps}, epochs={args.epochs})")
        train_humanmac(
            windows, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            n_coeff=args.n_coeff, d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, timesteps=args.timesteps, ddim_steps=args.ddim_steps,
        )
        return
    if args.model == "scene-simlpe":
        windows = build_or_load_scene_windows(args.datasets, args.stepsize, test_json)
    else:
        windows = build_or_load_windows(args.datasets, args.stepsize, test_json)
    n_windows = len(windows[0]) if args.model == "scene-simlpe" else len(windows)
    print(f"training on {n_windows} windows")
    train(
        windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vel_weight=args.vel_weight,
        horizon_floor=args.horizon_floor,
        n_blocks=args.n_blocks,
        output_mode=args.output_mode,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
