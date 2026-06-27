# Horizon-Weighted Loss + Tunable Training Knobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gentle linear horizon weighting to the forecasting loss and expose it (plus the already-existing-but-unwired `vel_weight`) as training knobs, so the next siMLPe retrain can beat the zero-velocity baseline (overall MPJPE 1.108).

**Architecture:** `forecasting/losses.py` gains a pure `horizon_weights(t, floor)` helper and optional `weights=` args on `mpjpe_loss`/`velocity_loss` (`weights=None` preserves today's plain-mean behavior). `forecasting/train.py` precomputes the two weight vectors once and threads them through `train_one_epoch`/`eval_loss`, and `main()` gains `--vel-weight` / `--horizon-floor` flags with new defaults (lr 5e-4, vel_weight 0.2, horizon_floor 0.2). No model, eval, or data changes — the scoreboard stays fair.

**Tech Stack:** Python, PyTorch, `unittest` (run via `python -m unittest`). venv at `/opt/study/.venv` (`source .venv/bin/activate`).

**Reference:** spec at `docs/superpowers/specs/2026-06-27-horizon-weighted-loss-design.md`.

---

## Context for the implementer

`forecasting/losses.py` today (the full file):

```python
import torch

from forecasting.config import N_JOINTS


def _per_joint_l2(pred, target):
    b, t, _ = pred.shape
    p = pred.reshape(b, t, N_JOINTS, 3)
    g = target.reshape(b, t, N_JOINTS, 3)
    return torch.linalg.norm(p - g, dim=-1)


def mpjpe_loss(pred, target):
    return _per_joint_l2(pred, target).mean()


def velocity_loss(pred, target):
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dg = target[:, 1:, :] - target[:, :-1, :]
    return _per_joint_l2(dp, dg).mean()
```

- `pred`/`target` are `[B, T, POSE_DIM]` where `POSE_DIM = 87` (= `N_JOINTS=29` × 3).
- `_per_joint_l2` returns a `[B, T, N_JOINTS]` curve of per-joint L2 distances.
- `velocity_loss` operates on first differences, so its time axis has length `T-1`.

The project has **no test files yet** for `forecasting/` (only `hik/data/utils.py` has unit tests, as a `unittest.TestCase` inside the module). We add a standalone `forecasting/test_losses.py` runnable with `python -m unittest forecasting.test_losses`.

Always `source .venv/bin/activate` before running anything.

---

## Task 1: `horizon_weights` helper

**Files:**
- Create: `forecasting/test_losses.py`
- Modify: `forecasting/losses.py`

- [ ] **Step 1: Write the failing test**

Create `forecasting/test_losses.py`:

```python
import unittest

import torch

from forecasting.losses import horizon_weights


class TestHorizonWeights(unittest.TestCase):
    def test_length_and_endpoints(self):
        w = horizon_weights(5, floor=0.2)
        self.assertEqual(w.shape, (5,))
        self.assertAlmostEqual(w[0].item(), 1.0, places=6)
        self.assertAlmostEqual(w[-1].item(), 0.2, places=6)

    def test_linear_midpoint(self):
        w = horizon_weights(5, floor=0.0)
        # k=2 of t=5 -> 1 - 1.0 * 2/4 = 0.5
        self.assertAlmostEqual(w[2].item(), 0.5, places=6)

    def test_floor_one_is_uniform(self):
        w = horizon_weights(7, floor=1.0)
        self.assertTrue(torch.allclose(w, torch.ones(7)))

    def test_monotonically_non_increasing(self):
        w = horizon_weights(10, floor=0.2)
        self.assertTrue(torch.all(w[1:] <= w[:-1] + 1e-9))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest forecasting.test_losses.TestHorizonWeights -v`
Expected: FAIL with `ImportError: cannot import name 'horizon_weights'`.

- [ ] **Step 3: Write minimal implementation**

In `forecasting/losses.py`, add after the imports (above `_per_joint_l2`):

```python
def horizon_weights(t, floor, device=None):
    """Linear per-frame weights: 1.0 at frame 0 down to `floor` at frame t-1.

    floor == 1.0 reproduces uniform weighting (all ones).
    """
    k = torch.arange(t, dtype=torch.float32, device=device)
    return 1.0 - (1.0 - floor) * (k / (t - 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest forecasting.test_losses.TestHorizonWeights -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add forecasting/losses.py forecasting/test_losses.py
git commit -m "feat(forecasting): add horizon_weights linear-decay helper"
```

---

## Task 2: weighted `mpjpe_loss`

**Files:**
- Modify: `forecasting/losses.py`
- Modify: `forecasting/test_losses.py`

- [ ] **Step 1: Write the failing test**

Append to `forecasting/test_losses.py` (before the `if __name__` block):

```python
from forecasting.losses import mpjpe_loss
from forecasting.config import POSE_DIM


class TestWeightedMpjpe(unittest.TestCase):
    def _pair(self, t=10):
        torch.manual_seed(0)
        target = torch.zeros(2, t, POSE_DIM)
        pred = torch.zeros(2, t, POSE_DIM)
        return pred, target

    def test_floor_one_equals_unweighted(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 10, POSE_DIM)
        target = torch.randn(2, 10, POSE_DIM)
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=1.0))
        self.assertAlmostEqual(plain.item(), weighted.item(), places=5)

    def test_late_error_downweighted(self):
        # error only in the LAST frame -> weighting should LOWER the loss
        pred, target = self._pair(t=10)
        pred[:, -1, :] = 1.0
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=0.2))
        self.assertLess(weighted.item(), plain.item())

    def test_early_error_upweighted(self):
        # error only in the FIRST frame -> weighting should RAISE the loss
        pred, target = self._pair(t=10)
        pred[:, 0, :] = 1.0
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=0.2))
        self.assertGreater(weighted.item(), plain.item())
```

Note: `horizon_weights` is already imported at the top of the file from Task 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest forecasting.test_losses.TestWeightedMpjpe -v`
Expected: FAIL — `mpjpe_loss() got an unexpected keyword argument 'weights'`.

- [ ] **Step 3: Write minimal implementation**

Replace `mpjpe_loss` in `forecasting/losses.py` with:

```python
def _weighted_time_mean(curve, weights):
    # curve: [B, T, N_JOINTS]; weights: [T] or None
    if weights is None:
        return curve.mean()
    per_frame = curve.mean(dim=(0, 2))            # [T]
    w = weights.to(per_frame.device)
    return (per_frame * w).sum() / w.sum()


def mpjpe_loss(pred, target, weights=None):
    return _weighted_time_mean(_per_joint_l2(pred, target), weights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest forecasting.test_losses -v`
Expected: PASS (all `TestHorizonWeights` + `TestWeightedMpjpe`).

- [ ] **Step 5: Commit**

```bash
git add forecasting/losses.py forecasting/test_losses.py
git commit -m "feat(forecasting): weighted mpjpe_loss with optional horizon weights"
```

---

## Task 3: weighted `velocity_loss`

**Files:**
- Modify: `forecasting/losses.py`
- Modify: `forecasting/test_losses.py`

- [ ] **Step 1: Write the failing test**

Append to `forecasting/test_losses.py` (before the `if __name__` block):

```python
from forecasting.losses import velocity_loss


class TestWeightedVelocity(unittest.TestCase):
    def test_floor_one_equals_unweighted(self):
        torch.manual_seed(2)
        pred = torch.randn(2, 10, POSE_DIM)
        target = torch.randn(2, 10, POSE_DIM)
        plain = velocity_loss(pred, target)
        # velocity curve length is T-1 = 9
        weighted = velocity_loss(pred, target, weights=horizon_weights(9, floor=1.0))
        self.assertAlmostEqual(plain.item(), weighted.item(), places=5)

    def test_late_error_downweighted(self):
        # a single late position jump -> velocity error concentrated at the end
        target = torch.zeros(2, 10, POSE_DIM)
        pred = torch.zeros(2, 10, POSE_DIM)
        pred[:, -1, :] = 1.0
        plain = velocity_loss(pred, target)
        weighted = velocity_loss(pred, target, weights=horizon_weights(9, floor=0.2))
        self.assertLess(weighted.item(), plain.item())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest forecasting.test_losses.TestWeightedVelocity -v`
Expected: FAIL — `velocity_loss() got an unexpected keyword argument 'weights'`.

- [ ] **Step 3: Write minimal implementation**

Replace `velocity_loss` in `forecasting/losses.py` with:

```python
def velocity_loss(pred, target, weights=None):
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dg = target[:, 1:, :] - target[:, :-1, :]
    return _weighted_time_mean(_per_joint_l2(dp, dg), weights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest forecasting.test_losses -v`
Expected: PASS (all three test classes).

- [ ] **Step 5: Commit**

```bash
git add forecasting/losses.py forecasting/test_losses.py
git commit -m "feat(forecasting): weighted velocity_loss with optional horizon weights"
```

---

## Task 4: thread weights through `train.py`

**Files:**
- Modify: `forecasting/train.py`

Current relevant code in `forecasting/train.py`:

```python
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
```

- [ ] **Step 1: Update the import**

In `forecasting/train.py` change:

```python
from forecasting.losses import mpjpe_loss, velocity_loss
```
to:
```python
from forecasting.losses import horizon_weights, mpjpe_loss, velocity_loss
```

- [ ] **Step 2: Add `pos_w`/`vel_w` params to the two loop functions**

Replace `train_one_epoch` with:

```python
def train_one_epoch(model, loader, optim, device, vel_weight=1.0, pos_w=None, vel_w=None):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        pred = model(x)
        loss = mpjpe_loss(pred, y, weights=pos_w) + vel_weight * velocity_loss(
            pred, y, weights=vel_w
        )
        loss.backward()
        optim.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)
```

Replace `eval_loss` with:

```python
@torch.no_grad()
def eval_loss(model, loader, device, vel_weight=1.0, pos_w=None, vel_w=None):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = mpjpe_loss(pred, y, weights=pos_w) + vel_weight * velocity_loss(
            pred, y, weights=vel_w
        )
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)
```

- [ ] **Step 3: Add `horizon_floor`, precompute weights, pass them in**

In the `train(...)` signature, change the defaults line `vel_weight=1.0,` block to add `horizon_floor` and lower `lr`/`vel_weight`. The current signature is:

```python
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
```

Replace it with:

```python
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
    seed=0,
    out_path="__default__",
):
```

Then, immediately after the optimizer/scheduler are created (after the line
`sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)`), add:

```python
    pos_w = horizon_weights(config.N_OUT, horizon_floor, device=device)
    vel_w = horizon_weights(config.N_OUT - 1, horizon_floor, device=device)
```

In the epoch loop, change the two calls from:

```python
        train_loss = train_one_epoch(model, train_loader, optim, device, vel_weight)
        val_loss = eval_loss(model, val_loader, device, vel_weight)
```
to:
```python
        train_loss = train_one_epoch(
            model, train_loader, optim, device, vel_weight, pos_w, vel_w
        )
        val_loss = eval_loss(model, val_loader, device, vel_weight, pos_w, vel_w)
```

- [ ] **Step 4: Verify training still imports and the weight shapes are right**

Run:
```bash
python -c "import torch; from forecasting.train import train_one_epoch, eval_loss; from forecasting.losses import horizon_weights; from forecasting import config; print(horizon_weights(config.N_OUT, 0.2).shape, horizon_weights(config.N_OUT-1, 0.2).shape)"
```
Expected: `torch.Size([250]) torch.Size([249])` and no import errors.

- [ ] **Step 5: Commit**

```bash
git add forecasting/train.py
git commit -m "feat(forecasting): thread horizon weights through training; new lr/vel defaults"
```

---

## Task 5: expose `--vel-weight` and `--horizon-floor` CLI flags

**Files:**
- Modify: `forecasting/train.py`

Current `main()`:

```python
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
```

- [ ] **Step 1: Add the two flags and lower the `--lr` default**

Replace the `parser.add_argument("--lr", ...)` line and add two flags so the parser block reads:

```python
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vel-weight", type=float, default=0.2)
    parser.add_argument("--horizon-floor", type=float, default=0.2)
    parser.add_argument("--n-blocks", type=int, default=4)
```

- [ ] **Step 2: Pass them into `train(...)`**

Replace the `train(...)` call in `main()` with:

```python
    train(
        windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vel_weight=args.vel_weight,
        horizon_floor=args.horizon_floor,
        n_blocks=args.n_blocks,
    )
```

- [ ] **Step 3: Verify the CLI parses**

Run: `python -m forecasting.train --help`
Expected: help text lists `--vel-weight`, `--horizon-floor`, and `--lr` (default 0.0005). No dataset is loaded by `--help`.

- [ ] **Step 4: Commit**

```bash
git add forecasting/train.py
git commit -m "feat(forecasting): expose --vel-weight and --horizon-floor CLI flags"
```

---

## Task 6: update the runbook

**Files:**
- Modify: `forecasting/README.md`

- [ ] **Step 1: Find the train command in the runbook**

Run: `grep -n "forecasting.train\|stepsize\|epochs" forecasting/README.md`
This locates the existing train invocation block to update.

- [ ] **Step 2: Update the train command and add a result slot**

Update the train command in `forecasting/README.md` to show the new flags (keep the surrounding prose/`export HIK_DATA` lines intact). The command should read:

```bash
python -m forecasting.train \
  --datasets A B C D --stepsize 50 --epochs 80 \
  --lr 5e-4 --vel-weight 0.2 --horizon-floor 0.2
```

Then, directly below the existing 2026-06-26 result table, add a new section:

```markdown
### Result (2026-06-27 retrain — horizon-weighted loss)

Config: lr 5e-4, vel_weight 0.2, horizon_floor 0.2 (gentle linear decay 1.0 → 0.2),
otherwise A+B+C+D, stepsize 50, 80 epochs. Beat-the-bar target: overall < 1.108.

| horizon | zero-velocity | siMLPe (weighted) |
|---------|--------------:|------------------:|
| overall | **1.108**     | TBD               |
| @1s     | **0.520**     | TBD               |
| @5s     | **1.254**     | TBD               |
| @10s    | **1.422**     | TBD               |
```

(The `TBD`s are intentional result placeholders to fill in *after* the VM retrain — they are data, not unfinished plan steps.)

- [ ] **Step 3: Commit**

```bash
git add forecasting/README.md
git commit -m "docs(forecasting): runbook for horizon-weighted retrain + result slot"
```

---

## Task 7: full test sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full forecasting loss suite**

Run: `python -m unittest forecasting.test_losses -v`
Expected: PASS, all classes (`TestHorizonWeights`, `TestWeightedMpjpe`, `TestWeightedVelocity`).

- [ ] **Step 2: Confirm the existing harness tests still pass**

Run: `python -m unittest hik.data.utils -v`
Expected: PASS (unchanged — we touched nothing under `hik/`).

- [ ] **Step 3: Confirm imports are clean**

Run: `python -c "import forecasting.train, forecasting.losses, forecasting.evaluate; print('ok')"`
Expected: `ok`.

No commit (verification only). The retrain itself runs on the GCP VM (kept stopped) per project setup — it is not part of this plan; this plan ends when the code + tests land.

---

## Notes for the implementer

- Heavy training/data runs OOM locally — do **not** run `python -m forecasting.train` against the real dataset on this machine. The unit tests and `--help`/import checks in this plan are all CPU-cheap and safe locally.
- `weights=None` everywhere preserves the old plain-mean behavior; the Task 2/3 "floor=1.0 equals unweighted" tests are the regression guard for that.
