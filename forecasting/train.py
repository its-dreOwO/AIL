import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from forecasting import config
from forecasting.data import WindowDataset, build_or_load_windows
from forecasting.losses import mpjpe_loss, velocity_loss
from forecasting.model import SiMLPe


def train_one_epoch(model, loader, optim, device, vel_weight=1.0):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        pred = model(x)
        loss = mpjpe_loss(pred, y) + vel_weight * velocity_loss(pred, y)
        loss.backward()
        optim.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device, vel_weight=1.0):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = mpjpe_loss(pred, y) + vel_weight * velocity_loss(pred, y)
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def train(
    windows,
    *,
    epochs,
    batch_size=256,
    lr=3e-3,
    val_frac=0.05,
    vel_weight=1.0,
    device=None,
    n_blocks=4,
    seed=0,
    out_path="__default__",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if out_path == "__default__":
        out_path = os.path.join(config.cache_dir(), "simlpe.pt")

    torch.manual_seed(seed)
    dataset = WindowDataset(windows)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SiMLPe(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    hist = {"train_loss": [], "val_loss": [], "best_val": float("inf"), "ckpt": None}
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optim, device, vel_weight)
        val_loss = eval_loss(model, val_loader, device, vel_weight)
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
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    parser.add_argument("--stepsize", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-blocks", type=int, default=4)
    args = parser.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    windows = build_or_load_windows(args.datasets, args.stepsize, test_json)
    print(f"training on {len(windows)} windows")
    train(
        windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_blocks=args.n_blocks,
    )


if __name__ == "__main__":
    main()
