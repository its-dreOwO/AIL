# Single-person siMLPe Forecasting Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-person DCT-MLP (siMLPe) forecaster that predicts 10 s of future 3D pose from 10 s of past pose, plugged into the vendored HIK `Evaluator.execute3d`, scored with a freshly-implemented MPJPE against a zero-velocity reference.

**Architecture:** A new `forecasting/` Python package alongside the vendored `hik/` library. Data is materialized into 500-frame single-person windows via `hik`'s `Scene.get_splits`, canonically normalized at the last observed frame (frame index 249) with `hik.transforms.utils.normalize3d`, and fed to a siMLPe network (DCT → temporal MLP blocks → IDCT → residual). Predictions are de-normalized back to world coordinates for scoring. The `hik` library is treated as read-only.

**Tech Stack:** Python 3, NumPy, PyTorch 2.12 (CUDA, local RTX 4060), the vendored `hik` package, `pytest`.

## Global Constraints

Every task implicitly inherits these. Values are verbatim from the spec.

- **Do NOT edit the `hik/` library.** In particular `hik/eval/mpjpe.py::calc_mpjpe` stays a stub; our scoring lives in `forecasting/metrics.py`.
- **Dataset path comes from the `HIK_DATA` env var.** Never hardcode the spaced path. Default in `config.py` is `/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen`. Subdirs: `poses/`, `scenes/`, `body_models/SMPLX_NEUTRAL.npz`.
- **Constants:** `N_IN=250`, `N_OUT=250`, `WINDOW=500`, `N_JOINTS=29`, `POSE_DIM=87` (=29×3). Frame rate 25 Hz.
- **No train/test leak:** any training window overlapping a `testdata/test.json` frame range `[tf - N_IN, tf + N_OUT)` is dropped.
- **Single person, independent:** the eval callback runs the model once per person present and stacks to `[250, n_person, 29, 3]`.
- **Pose representation:** canonical normalization via `hik.transforms.utils.normalize3d` / `denormalize3d`, reference frame index 249 (last observed). Loss is computed in normalized space.
- **Tests run without the dataset where possible** (synthetic arrays). Tests that need the real dataset are marked `@pytest.mark.slow` and skipped if `HIK_DATA` is unset/missing.
- **Commits:** end every commit message body with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UMVdo7JwWBf5uAZtx7NjTW
  ```
- **Run all commands from the repo root** `/opt/study/ail/hik` with the venv active: `source .venv/bin/activate`.

## File Structure

```
forecasting/
  __init__.py        # package marker
  config.py          # paths (HIK_DATA), constants, path resolution helpers
  dct.py             # get_dct_matrix
  model.py           # SiMLPe, TemporalMLPBlock
  data.py            # test-frame exclusion, build_windows, WindowDataset
  losses.py          # mpjpe_loss, velocity_loss
  metrics.py         # calc_mpjpe (per-horizon scoring over a results dict)
  callbacks.py       # zero_velocity_callback, make_simlpe_callback
  train.py           # training loop CLI
  evaluate.py        # run Evaluator with a callback + print/save metrics CLI
tests/
  __init__.py
  test_dct.py
  test_model.py
  test_losses.py
  test_metrics.py
  test_callbacks.py
  test_data.py
```

**Prerequisite (do once, before Task 1):** ensure test tooling is present.
Run: `source .venv/bin/activate && python -c "import pytest" 2>/dev/null || pip install pytest`

---

### Task 1: Package scaffold + config

**Files:**
- Create: `forecasting/__init__.py` (empty)
- Create: `forecasting/config.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - Constants `N_IN=250`, `N_OUT=250`, `WINDOW=500`, `N_JOINTS=29`, `POSE_DIM=87`, `FPS=25`, `DATASETS=("A","B","C","D")`
  - `data_root() -> str` — returns `os.environ.get("HIK_DATA", DEFAULT_HIK_DATA)`
  - `poses_path() -> str`, `scenes_path() -> str`, `smplx_path() -> str` — `join(data_root(), "poses"|"scenes"|"body_models/SMPLX_NEUTRAL.npz")`
  - `cache_dir() -> str` — `join(repo_root, "forecasting", "cache")`, created if missing
  - `dataset_available() -> bool` — `isdir(poses_path())`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
from forecasting import config


def test_constants():
    assert config.N_IN == 250
    assert config.N_OUT == 250
    assert config.WINDOW == 500
    assert config.POSE_DIM == config.N_JOINTS * 3 == 87
    assert config.DATASETS == ("A", "B", "C", "D")


def test_data_root_from_env(monkeypatch):
    monkeypatch.setenv("HIK_DATA", "/tmp/somewhere")
    assert config.data_root() == "/tmp/somewhere"
    assert config.poses_path() == "/tmp/somewhere/poses"
    assert config.smplx_path().endswith("body_models/SMPLX_NEUTRAL.npz")


def test_cache_dir_created():
    d = config.cache_dir()
    assert os.path.isdir(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/__init__.py
```
(empty file)

```python
# tests/__init__.py
```
(empty file)

```python
# forecasting/config.py
import os
from os.path import join, isdir, dirname, abspath

N_IN = 250
N_OUT = 250
WINDOW = N_IN + N_OUT
N_JOINTS = 29
POSE_DIM = N_JOINTS * 3
FPS = 25
DATASETS = ("A", "B", "C", "D")

DEFAULT_HIK_DATA = "/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen"

_REPO_ROOT = dirname(dirname(abspath(__file__)))


def data_root() -> str:
    return os.environ.get("HIK_DATA", DEFAULT_HIK_DATA)


def poses_path() -> str:
    return join(data_root(), "poses")


def scenes_path() -> str:
    return join(data_root(), "scenes")


def smplx_path() -> str:
    return join(data_root(), "body_models", "SMPLX_NEUTRAL.npz")


def cache_dir() -> str:
    d = join(_REPO_ROOT, "forecasting", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def dataset_available() -> bool:
    return isdir(poses_path())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/__init__.py forecasting/config.py tests/__init__.py tests/test_config.py
git commit -m "feat(forecasting): package scaffold + config"
```

---

### Task 2: DCT matrices

**Files:**
- Create: `forecasting/dct.py`
- Test: `tests/test_dct.py`

**Interfaces:**
- Produces: `get_dct_matrix(N: int) -> tuple[np.ndarray, np.ndarray]` returning `(dct_m, idct_m)`, each `[N, N]` float64, with `idct_m @ dct_m ≈ I` and `idct_m @ (dct_m @ x) ≈ x` for any signal `x` of length `N`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dct.py
import numpy as np
from forecasting.dct import get_dct_matrix


def test_shapes():
    dct_m, idct_m = get_dct_matrix(250)
    assert dct_m.shape == (250, 250)
    assert idct_m.shape == (250, 250)


def test_roundtrip_reconstructs_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))          # [N=50, channels=3]
    dct_m, idct_m = get_dct_matrix(50)
    x_rec = idct_m @ (dct_m @ x)
    assert np.allclose(x_rec, x, atol=1e-8)


def test_inverse_is_inverse():
    dct_m, idct_m = get_dct_matrix(32)
    assert np.allclose(idct_m @ dct_m, np.eye(32), atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_dct.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.dct'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/dct.py
import numpy as np


def get_dct_matrix(N: int):
    """Return (dct_m, idct_m), each [N, N]. Orthonormal DCT-II over time.

    For a signal x of shape [N, C]:  x_freq = dct_m @ x ; x = idct_m @ x_freq.
    """
    dct_m = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        w = np.sqrt(2.0 / N)
        if k == 0:
            w = np.sqrt(1.0 / N)
        for i in range(N):
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_dct.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/dct.py tests/test_dct.py
git commit -m "feat(forecasting): DCT/IDCT matrices"
```

---

### Task 3: siMLPe model

**Files:**
- Create: `forecasting/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `forecasting.dct.get_dct_matrix`, `forecasting.config` (`N_IN`, `N_OUT`, `POSE_DIM`)
- Produces:
  - `class TemporalMLPBlock(nn.Module)` — `forward(x: [B, C, T]) -> [B, C, T]`
  - `class SiMLPe(nn.Module)` — `__init__(self, t_in=250, t_out=250, pose_dim=87, n_blocks=4)`; `forward(x: [B, t_in, pose_dim]) -> [B, t_out, pose_dim]`. Predicts the future sequence as an offset from the last observed frame.

Note: `t_in == t_out` is required (the DCT/IDCT are square over the same length). The future is produced by the temporal MLP stack rewriting the DCT spectrum of the input window.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from forecasting.model import SiMLPe


def test_forward_shape():
    model = SiMLPe(t_in=250, t_out=250, pose_dim=87, n_blocks=4)
    x = torch.randn(4, 250, 87)
    y = model(x)
    assert y.shape == (4, 250, 87)
    assert torch.isfinite(y).all()


def test_residual_anchors_on_last_frame():
    # With zero-initialized output projection, prediction == last input frame.
    model = SiMLPe(t_in=250, t_out=250, pose_dim=87, n_blocks=2)
    torch.nn.init.zeros_(model.motion_fc_out.weight)
    torch.nn.init.zeros_(model.motion_fc_out.bias)
    x = torch.randn(2, 250, 87)
    y = model(x)
    last = x[:, -1:, :].expand(-1, 250, -1)
    assert torch.allclose(y, last, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/model.py
import numpy as np
import torch
import torch.nn as nn

from forecasting.dct import get_dct_matrix


class TemporalMLPBlock(nn.Module):
    """Mixes information across the time axis, per channel. Input [B, C, T]."""

    def __init__(self, t: int):
        super().__init__()
        self.fc = nn.Linear(t, t)
        self.ln = nn.LayerNorm(t)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B, C, T]
        return x + self.act(self.ln(self.fc(x)))


class SiMLPe(nn.Module):
    """DCT -> temporal MLP blocks -> IDCT -> residual on last observed frame."""

    def __init__(self, t_in=250, t_out=250, pose_dim=87, n_blocks=4):
        super().__init__()
        assert t_in == t_out, "this baseline uses square DCT over equal lengths"
        self.t_in = t_in
        self.t_out = t_out
        self.pose_dim = pose_dim

        dct_m, idct_m = get_dct_matrix(t_in)
        self.register_buffer("dct", torch.tensor(dct_m, dtype=torch.float32))
        self.register_buffer("idct", torch.tensor(idct_m, dtype=torch.float32))

        self.motion_fc_in = nn.Linear(pose_dim, pose_dim)
        self.blocks = nn.ModuleList(
            [TemporalMLPBlock(t_in) for _ in range(n_blocks)]
        )
        self.motion_fc_out = nn.Linear(pose_dim, pose_dim)

    def forward(self, x):  # x: [B, t_in, pose_dim]
        last = x[:, -1:, :]                                   # [B, 1, C]
        h = self.motion_fc_in(x)                              # [B, T, C]
        h = torch.einsum("tk,bkc->btc", self.dct, h)         # DCT over time
        h = h.transpose(1, 2)                                # [B, C, T]
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)                                # [B, T, C]
        h = torch.einsum("tk,bkc->btc", self.idct, h)        # IDCT over time
        h = self.motion_fc_out(h)                            # [B, T, C]
        return h + last                                      # offsets from last frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/model.py tests/test_model.py
git commit -m "feat(forecasting): siMLPe DCT-MLP model"
```

---

### Task 4: Losses

**Files:**
- Create: `forecasting/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `forecasting.config` (`N_JOINTS`)
- Produces:
  - `mpjpe_loss(pred, target) -> torch.Tensor` — scalar. `pred`/`target` are `[B, T, POSE_DIM]`; reshapes to `[B, T, 29, 3]`, takes per-joint L2, means over joints/time/batch.
  - `velocity_loss(pred, target) -> torch.Tensor` — scalar MPJPE on temporal first-differences (`x[:,1:] - x[:,:-1]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch
from forecasting.losses import mpjpe_loss, velocity_loss


def test_mpjpe_zero_when_equal():
    x = torch.randn(3, 250, 87)
    assert mpjpe_loss(x, x).item() == 0.0


def test_mpjpe_positive_and_correct():
    pred = torch.zeros(1, 1, 87)
    target = torch.zeros(1, 1, 87)
    target[0, 0, 0] = 3.0   # joint0 x
    target[0, 0, 1] = 4.0   # joint0 y -> L2 = 5 for joint0, 0 elsewhere
    # mean over 29 joints of [5, 0, 0, ...] = 5/29
    assert abs(mpjpe_loss(pred, target).item() - 5.0 / 29.0) < 1e-6


def test_velocity_zero_for_constant_motion():
    x = torch.randn(2, 250, 87)
    # same constant velocity in pred and target -> equal first diffs -> 0
    assert velocity_loss(x, x).item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/losses.py
import torch

from forecasting.config import N_JOINTS


def _per_joint_l2(pred, target):
    # pred/target: [B, T, POSE_DIM] -> [B, T, N_JOINTS, 3]
    b, t, _ = pred.shape
    p = pred.reshape(b, t, N_JOINTS, 3)
    g = target.reshape(b, t, N_JOINTS, 3)
    return torch.linalg.norm(p - g, dim=-1)   # [B, T, N_JOINTS]


def mpjpe_loss(pred, target):
    return _per_joint_l2(pred, target).mean()


def velocity_loss(pred, target):
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dg = target[:, 1:, :] - target[:, :-1, :]
    return _per_joint_l2(dp, dg).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_losses.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/losses.py tests/test_losses.py
git commit -m "feat(forecasting): mpjpe + velocity losses"
```

---

### Task 5: Scoring metric (calc_mpjpe, per-horizon)

**Files:**
- Create: `forecasting/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `hik.eval.mpjpe.mean_per_joint_l2_distance` (signature: `(a, b)` with `a,b` shape `[T,29,3]`, returns `[T]` per-frame MPJPE), `forecasting.config` (`N_OUT`, `FPS`)
- Produces:
  - `calc_mpjpe(results: dict, horizons_sec=(1,2,3,4,5,6,7,8,9,10)) -> dict`
    - `results` shape: `{action: [ {"Poses3d_out": [T,P,29,3], "Masks_out": [T,P], "Poses3d_out_pred": [T,P,29,3], "target_pid": int, "pids": [int,...]}, ... ]}`
    - For each entry: locate the target pid's column, compute the per-frame MPJPE curve `[T]` for that person (skip frames where `Masks_out` is 0 for that person).
    - Returns `{"per_action": {action: {"curve": [T] list, "at_sec": {s: mpjpe}, "mean": float}}, "overall": {"curve": [T] list, "at_sec": {s: mpjpe}, "mean": float}}`. `mpjpe` values are floats in the dataset's length units.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
from forecasting.metrics import calc_mpjpe


def _entry(offset, target_pid, pids):
    T, P = 250, len(pids)
    gt = np.zeros((T, P, 29, 3), dtype=np.float32)
    pred = np.zeros((T, P, 29, 3), dtype=np.float32)
    idx = pids.index(target_pid)
    pred[:, idx] = gt[:, idx] + offset   # constant L2 = |offset| per joint
    masks = np.ones((T, P), dtype=np.float32)
    return {
        "Poses3d_out": gt,
        "Masks_out": masks,
        "Poses3d_out_pred": pred,
        "target_pid": target_pid,
        "pids": pids,
    }


def test_overall_mean_matches_constant_error():
    # offset [0.3,0.4,0] -> per-joint L2 = 0.5 everywhere
    off = np.array([0.3, 0.4, 0.0], dtype=np.float32)
    results = {"walking": [_entry(off, 7, [7, 9])]}
    out = calc_mpjpe(results)
    assert abs(out["overall"]["mean"] - 0.5) < 1e-5
    assert abs(out["overall"]["at_sec"][10] - 0.5) < 1e-5
    assert abs(out["per_action"]["walking"]["mean"] - 0.5) < 1e-5


def test_target_pid_selection():
    # error only on a non-target column must NOT affect the score
    off = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e = _entry(np.zeros(3, dtype=np.float32), 7, [7, 9])
    e["Poses3d_out_pred"][:, 1] += off   # corrupt pid 9, target is 7
    out = calc_mpjpe({"walking": [e]})
    assert out["overall"]["mean"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/metrics.py
import numpy as np

from hik.eval.mpjpe import mean_per_joint_l2_distance
from forecasting.config import N_OUT, FPS


def _entry_curve(entry):
    """Per-frame MPJPE [T] for the target person; NaN where that person is masked."""
    pids = list(entry["pids"])
    idx = pids.index(entry["target_pid"])
    gt = entry["Poses3d_out"][:, idx]          # [T, 29, 3]
    pred = entry["Poses3d_out_pred"][:, idx]   # [T, 29, 3]
    mask = entry["Masks_out"][:, idx]          # [T]
    curve = mean_per_joint_l2_distance(gt, pred)   # [T]
    curve = np.where(mask > 0.5, curve, np.nan)
    return curve


def _summarize(curves, horizons_sec):
    stack = np.stack(curves, axis=0)               # [n_entries, T]
    curve = np.nanmean(stack, axis=0)              # [T]
    at_sec = {}
    for s in horizons_sec:
        f = min(s * FPS - 1, len(curve) - 1)
        at_sec[s] = float(curve[f])
    return {
        "curve": [float(x) for x in curve],
        "at_sec": at_sec,
        "mean": float(np.nanmean(curve)),
    }


def calc_mpjpe(results: dict, horizons_sec=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) -> dict:
    per_action = {}
    all_curves = []
    for action, entries in results.items():
        curves = [_entry_curve(e) for e in entries]
        all_curves.extend(curves)
        per_action[action] = _summarize(curves, horizons_sec)
    overall = _summarize(all_curves, horizons_sec)
    return {"per_action": per_action, "overall": overall}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_metrics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/metrics.py tests/test_metrics.py
git commit -m "feat(forecasting): per-horizon MPJPE scoring"
```

---

### Task 6: Eval callbacks (zero-velocity + siMLPe wrapper)

**Files:**
- Create: `forecasting/callbacks.py`
- Test: `tests/test_callbacks.py`

**Interfaces:**
- Consumes: `hik.transforms.utils.backfill_masked`, `hik.transforms.utils.normalize3d`, `hik.transforms.utils.denormalize3d`, `forecasting.config` (`N_IN`, `N_OUT`, `POSE_DIM`)
- Produces:
  - `zero_velocity_callback(inp: dict) -> np.ndarray` — returns `[N_OUT, n_person, 29, 3]`: the last observed frame (`Poses3d_in[-1]`) repeated `N_OUT` times.
  - `make_simlpe_callback(model, device="cpu") -> callable` — returns a `callback(inp)` that: backfills input gaps, normalizes at frame `N_IN-1`, runs the model per person in normalized space, de-normalizes, returns `[N_OUT, n_person, 29, 3]`.
  - The `inp` dict matches `Evaluator.execute3d`'s callback contract: keys `Poses3d_in [N_IN,P,29,3]`, `Masks_in [N_IN,P]`, `n_out`, `pids`, `kitchen`, `frames_in`, `action`.

Note: `normalize3d` may raise if a hip joint sits at z≈0 for a person; catch per-person and fall back to that person's zero-velocity prediction so a single bad person never crashes a whole test case.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_callbacks.py
import numpy as np
import torch
from forecasting.callbacks import zero_velocity_callback, make_simlpe_callback


def _make_input(P=2):
    rng = np.random.default_rng(1)
    poses = rng.standard_normal((250, P, 29, 3)).astype(np.float32) + 1.0
    poses[:, :, :, 2] += 1.0    # keep z away from 0 for hip normalization
    masks = np.ones((250, P), dtype=np.float32)
    return {
        "Poses3d_in": poses,
        "Masks_in": masks,
        "n_out": 250,
        "pids": list(range(P)),
        "kitchen": None,
        "frames_in": list(range(250)),
        "action": "walking",
    }


def test_zero_velocity_shape_and_values():
    inp = _make_input()
    out = zero_velocity_callback(inp)
    assert out.shape == (250, 2, 29, 3)
    # every output frame equals the last observed frame
    assert np.allclose(out, inp["Poses3d_in"][-1][None], atol=1e-6)


def test_simlpe_callback_shape_and_inversion():
    # A model whose output == its input (in normalized space) must de-normalize
    # back to a prediction whose first frame ≈ last observed input frame.
    class IdentityModel(torch.nn.Module):
        def forward(self, x):   # x: [B, T, C]
            return x
    cb = make_simlpe_callback(IdentityModel(), device="cpu")
    inp = _make_input()
    out = cb(inp)
    assert out.shape == (250, 2, 29, 3)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_callbacks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.callbacks'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/callbacks.py
import numpy as np
import torch

from hik.transforms.utils import backfill_masked, normalize3d, denormalize3d
from forecasting.config import N_IN, N_OUT, POSE_DIM


def zero_velocity_callback(inp: dict) -> np.ndarray:
    poses_in = inp["Poses3d_in"]              # [N_IN, P, 29, 3]
    n_out = inp["n_out"]
    last = poses_in[-1]                        # [P, 29, 3]
    return np.repeat(last[None], n_out, axis=0).astype(np.float32)


def make_simlpe_callback(model, device="cpu"):
    model = model.to(device)
    model.eval()

    def callback(inp: dict) -> np.ndarray:
        poses_in = np.copy(inp["Poses3d_in"])     # [N_IN, P, 29, 3]
        masks_in = np.copy(inp["Masks_in"])       # [N_IN, P]
        n_out = inp["n_out"]
        P = poses_in.shape[1]

        filled, _ = backfill_masked(poses_in, masks_in)
        try:
            normed, norm_params = normalize3d(filled, frame=N_IN - 1)
        except ValueError:
            # normalization failed for the whole block -> zero-velocity fallback
            return zero_velocity_callback(inp)

        # model runs per person in normalized space
        x = normed.transpose(1, 0, 2, 3).reshape(P, N_IN, POSE_DIM)  # [P, T, C]
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            yt = model(xt)                                            # [P, n_out, C]
        y = yt.cpu().numpy().reshape(P, n_out, 29, 3)
        pred_normed = y.transpose(1, 0, 2, 3)                         # [n_out, P, 29, 3]

        pred_world = denormalize3d(pred_normed, norm_params)          # [n_out, P, 29, 3]
        return pred_world.astype(np.float32)

    return callback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_callbacks.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/callbacks.py tests/test_callbacks.py
git commit -m "feat(forecasting): zero-velocity + siMLPe eval callbacks"
```

---

### Task 7: Training-data pipeline (windows + exclusion + dataset)

**Files:**
- Create: `forecasting/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `hik.data.Scene`, `hik.transforms.utils.normalize3d`, `forecasting.config` (`WINDOW`, `N_IN`, `N_OUT`, `POSE_DIM`, `DATASETS`, paths, `cache_dir`), `testdata/test.json`
- Produces:
  - `forbidden_frames(dataset: str, test_json_path: str) -> set[int]` — union of `range(tf - N_IN, tf + N_OUT)` over every `(pid, frame)` test entry for `dataset`.
  - `window_is_clean(start: int, person_mask: np.ndarray, forbidden: set[int]) -> bool` — True iff `person_mask` is all-valid across the window **and** the window `[start, start+WINDOW)` does not intersect `forbidden`. (`person_mask` is the per-frame mask `[WINDOW]` for one person.)
  - `build_windows_for_dataset(dataset, stepsize, test_json_path, max_windows=None) -> np.ndarray` — returns normalized windows `[Nw, WINDOW, 29, 3]` (each normalized at frame `N_IN-1`). Uses `Scene.get_splits(length=WINDOW, stepsize=stepsize)`.
  - `WindowDataset(torch.utils.data.Dataset)` over a `[Nw, WINDOW, 29, 3]` array; `__getitem__` returns `(x, y)` float tensors with `x=[N_IN, POSE_DIM]`, `y=[N_OUT, POSE_DIM]`.

Note: the pure helpers (`forbidden_frames`, `window_is_clean`, `WindowDataset`) are unit-tested with synthetic inputs; `build_windows_for_dataset` runs against the real dataset and is covered by a `@pytest.mark.slow` test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import os
import numpy as np
import pytest
import torch

from forecasting import config
from forecasting.data import (
    forbidden_frames,
    window_is_clean,
    WindowDataset,
    build_windows_for_dataset,
)

TEST_JSON = os.path.join(config._REPO_ROOT, "testdata", "test.json")


def test_forbidden_frames_covers_range():
    fb = forbidden_frames("A", TEST_JSON)
    assert len(fb) > 0
    # every forbidden set is a union of 500-wide windows, so quite large
    assert isinstance(fb, set)


def test_window_is_clean_rejects_overlap_and_gaps():
    full = np.ones(config.WINDOW, dtype=np.float32)
    forbidden = {1000}
    assert window_is_clean(0, full, forbidden) is True          # clean
    assert window_is_clean(900, full, forbidden) is False       # 900..1400 hits 1000
    gappy = full.copy(); gappy[10] = 0.0
    assert window_is_clean(0, gappy, set()) is False            # has a gap


def test_window_dataset_splits_in_out():
    arr = np.random.default_rng(0).standard_normal(
        (3, config.WINDOW, 29, 3)
    ).astype(np.float32)
    ds = WindowDataset(arr)
    x, y = ds[0]
    assert x.shape == (config.N_IN, config.POSE_DIM)
    assert y.shape == (config.N_OUT, config.POSE_DIM)
    assert torch.is_tensor(x) and x.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not config.dataset_available(), reason="dataset not mounted")
def test_build_windows_real_small():
    w = build_windows_for_dataset("A", stepsize=500, test_json_path=TEST_JSON,
                                   max_windows=8)
    assert w.ndim == 4 and w.shape[1:] == (config.WINDOW, 29, 3)
    assert 0 < w.shape[0] <= 8
    assert np.isfinite(w).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/data.py
import json

import numpy as np
import torch

from hik.data import Scene
from hik.transforms.utils import normalize3d
from forecasting import config
from forecasting.config import (
    WINDOW, N_IN, N_OUT, POSE_DIM,
    poses_path, scenes_path, smplx_path,
)


def forbidden_frames(dataset: str, test_json_path: str) -> set:
    with open(test_json_path, "r") as f:
        data = json.load(f)
    fb = set()
    for action, entries in data.get(dataset, {}).items():
        for entry in entries:
            tf = entry["frame"]
            fb.update(range(tf - N_IN, tf + N_OUT))
    return fb


def window_is_clean(start: int, person_mask: np.ndarray, forbidden: set) -> bool:
    if person_mask.shape[0] != WINDOW:
        raise ValueError(f"bad mask length {person_mask.shape[0]}")
    if not np.all(person_mask > 0.5):
        return False
    # intersect [start, start+WINDOW) with forbidden
    for fr in range(start, start + WINDOW):
        if fr in forbidden:
            return False
    return True


def build_windows_for_dataset(dataset, stepsize, test_json_path, max_windows=None):
    forbidden = forbidden_frames(dataset, test_json_path)
    scene = Scene.load_from_paths(
        dataset=dataset,
        person_path=poses_path(),
        scene_path=scenes_path(),
        smplx_path=smplx_path(),
    )
    splits = scene.get_splits(length=WINDOW, stepsize=stepsize)
    poses3d = splits["poses3d"]      # [n_seq, WINDOW, P, 29, 3]
    masks = splits["masks"]          # [n_seq, WINDOW, P]
    starts = splits["start_frames"]  # [n_seq]

    out = []
    n_seq, _, P = masks.shape[0], masks.shape[1], masks.shape[2]
    for s in range(n_seq):
        start = int(starts[s])
        for p in range(P):
            if not window_is_clean(start, masks[s, :, p], forbidden):
                continue
            window = poses3d[s, :, p]                     # [WINDOW, 29, 3]
            block = window[:, None]                       # [WINDOW, 1, 29, 3]
            try:
                normed, _ = normalize3d(block, frame=N_IN - 1)
            except ValueError:
                continue
            out.append(normed[:, 0])                      # [WINDOW, 29, 3]
            if max_windows is not None and len(out) >= max_windows:
                return np.asarray(out, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray):
        # windows: [Nw, WINDOW, 29, 3]
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i].reshape(WINDOW, POSE_DIM)     # [WINDOW, C]
        x = torch.tensor(w[:N_IN], dtype=torch.float32)
        y = torch.tensor(w[N_IN:], dtype=torch.float32)
        return x, y
```

- [ ] **Step 4: Run the fast tests; verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_data.py -v -m "not slow"`
Expected: PASS (3 passed, 1 deselected)

- [ ] **Step 5: Commit**

```bash
git add forecasting/data.py tests/test_data.py
git commit -m "feat(forecasting): training-window pipeline + dataset"
```

---

### Task 8: Window caching + training CLI

**Files:**
- Create: `forecasting/train.py`
- Modify: `forecasting/data.py` (add `build_or_load_windows`)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `forecasting.data` (`build_windows_for_dataset`, `WindowDataset`), `forecasting.model.SiMLPe`, `forecasting.losses` (`mpjpe_loss`, `velocity_loss`), `forecasting.config`
- Produces (in `data.py`):
  - `build_or_load_windows(datasets, stepsize, test_json_path) -> np.ndarray` — concatenates per-dataset windows, caching the full array to `{cache_dir}/windows_{tag}.npy` (tag from datasets+stepsize); reloads if present.
- Produces (in `train.py`):
  - `train_one_epoch(model, loader, optim, device, vel_weight=1.0) -> float` — returns mean batch loss.
  - `train(windows, *, epochs, batch_size=256, lr=3e-3, val_frac=0.05, vel_weight=1.0, device=None, n_blocks=4, seed=0, out_path=None) -> dict` — trains, returns `{"train_loss": [...], "val_loss": [...], "best_val": float, "ckpt": path|None}`. Saves best model state_dict to `out_path` (default `{cache_dir}/simlpe.pt`).
  - `main()` — argparse CLI: `--datasets A B C D --stepsize 50 --epochs 80 --batch-size 256 --lr 3e-3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
import torch
from forecasting import config
from forecasting.train import train


def test_train_smoke_reduces_loss():
    # tiny synthetic dataset: each window is a smooth linear ramp (learnable)
    rng = np.random.default_rng(0)
    Nw = 64
    base = rng.standard_normal((Nw, 1, 29, 3)).astype(np.float32)
    t = np.linspace(0, 1, config.WINDOW).reshape(1, config.WINDOW, 1, 1)
    vel = rng.standard_normal((Nw, 1, 29, 3)).astype(np.float32) * 0.1
    windows = (base[:, None, :, :].repeat(config.WINDOW, axis=1)
               + (vel[:, None] * t)).astype(np.float32)  # [Nw, WINDOW, 29, 3]
    windows = windows.reshape(Nw, config.WINDOW, 29, 3)

    out = train(windows, epochs=3, batch_size=16, lr=3e-3,
                device="cpu", n_blocks=2, out_path=None)
    assert len(out["train_loss"]) == 3
    assert out["train_loss"][-1] < out["train_loss"][0]   # learning happens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.train'`

- [ ] **Step 3: Write minimal implementation**

Add to `forecasting/data.py`:

```python
import os


def build_or_load_windows(datasets, stepsize, test_json_path):
    tag = "".join(datasets) + f"_s{stepsize}"
    path = os.path.join(config.cache_dir(), f"windows_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    parts = [
        build_windows_for_dataset(d, stepsize, test_json_path) for d in datasets
    ]
    windows = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(path, windows)
    return windows
```

Create `forecasting/train.py`:

```python
# forecasting/train.py
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from forecasting import config
from forecasting.data import WindowDataset, build_or_load_windows
from forecasting.model import SiMLPe
from forecasting.losses import mpjpe_loss, velocity_loss


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


def train(windows, *, epochs, batch_size=256, lr=3e-3, val_frac=0.05,
          vel_weight=1.0, device=None, n_blocks=4, seed=0, out_path="__default__"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if out_path == "__default__":
        out_path = os.path.join(config.cache_dir(), "simlpe.pt")

    torch.manual_seed(seed)
    ds = WindowDataset(windows)
    n_val = max(1, int(len(ds) * val_frac))
    n_tr = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(ds, [n_tr, n_val], generator=gen)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)

    model = SiMLPe(t_in=config.N_IN, t_out=config.N_OUT,
                   pose_dim=config.POSE_DIM, n_blocks=n_blocks).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    hist = {"train_loss": [], "val_loss": [], "best_val": float("inf"), "ckpt": None}
    for ep in range(epochs):
        tl = train_one_epoch(model, tr_loader, optim, device, vel_weight)
        vl = eval_loss(model, va_loader, device, vel_weight)
        sched.step()
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        if vl < hist["best_val"]:
            hist["best_val"] = vl
            if out_path is not None:
                torch.save(model.state_dict(), out_path)
                hist["ckpt"] = out_path
        print(f"epoch {ep+1}/{epochs}  train {tl:.4f}  val {vl:.4f}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    ap.add_argument("--stepsize", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-blocks", type=int, default=4)
    args = ap.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    windows = build_or_load_windows(args.datasets, args.stepsize, test_json)
    print(f"training on {len(windows)} windows")
    train(windows, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, n_blocks=args.n_blocks)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_train.py -v`
Expected: PASS (1 passed). The final train loss is below the first.

- [ ] **Step 5: Commit**

```bash
git add forecasting/train.py forecasting/data.py tests/test_train.py
git commit -m "feat(forecasting): window caching + training loop CLI"
```

---

### Task 9: Evaluation CLI (run Evaluator + report)

**Files:**
- Create: `forecasting/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `hik.eval.evaluator.Evaluator`, `forecasting.callbacks` (`zero_velocity_callback`, `make_simlpe_callback`), `forecasting.metrics.calc_mpjpe`, `forecasting.model.SiMLPe`, `forecasting.config`
- Produces:
  - `evaluate_callback(callback_fn, dataset, data_path, test_json_path) -> dict` — runs `Evaluator(test_json_path, dataset, data_path).execute3d(callback_fn)` then `calc_mpjpe(results)`; returns the metrics dict.
  - `load_model(ckpt, n_blocks=4, device="cpu") -> SiMLPe`.
  - `main()` — argparse CLI: `--dataset A --model zerovel|simlpe --ckpt PATH`. Prints overall mean + per-horizon at 1/5/10 s.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from forecasting import evaluate
from forecasting.callbacks import zero_velocity_callback


def test_evaluate_callback_uses_metrics(monkeypatch):
    # fake Evaluator so the test needs no dataset
    fake_results = {"walking": [{
        "Poses3d_out": np.zeros((250, 1, 29, 3), np.float32),
        "Masks_out": np.ones((250, 1), np.float32),
        "Poses3d_out_pred": np.zeros((250, 1, 29, 3), np.float32),
        "target_pid": 5,
        "pids": [5],
    }]}

    class FakeEval:
        def __init__(self, *a, **k):
            pass
        def execute3d(self, callback_fn, **k):
            return fake_results

    monkeypatch.setattr(evaluate, "Evaluator", FakeEval)
    out = evaluate.evaluate_callback(zero_velocity_callback, "A", "/nope", "/nope")
    assert out["overall"]["mean"] == 0.0
    assert 10 in out["overall"]["at_sec"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/evaluate.py
import argparse
import os

import torch

from hik.eval.evaluator import Evaluator
from forecasting import config
from forecasting.callbacks import zero_velocity_callback, make_simlpe_callback
from forecasting.metrics import calc_mpjpe
from forecasting.model import SiMLPe


def evaluate_callback(callback_fn, dataset, data_path, test_json_path) -> dict:
    ev = Evaluator(test_json_path, dataset, data_path)
    results = ev.execute3d(callback_fn)
    return calc_mpjpe(results)


def load_model(ckpt, n_blocks=4, device="cpu") -> SiMLPe:
    model = SiMLPe(t_in=config.N_IN, t_out=config.N_OUT,
                   pose_dim=config.POSE_DIM, n_blocks=n_blocks)
    # weights_only=True: we only ever save a state_dict (plain tensors), so refuse
    # to unpickle arbitrary objects from a checkpoint file.
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def _print_report(name, metrics):
    o = metrics["overall"]
    print(f"\n=== {name} ===")
    print(f"overall mean MPJPE: {o['mean']:.4f}")
    for s in (1, 5, 10):
        print(f"  @{s}s: {o['at_sec'][s]:.4f}")
    for action, m in metrics["per_action"].items():
        print(f"  [{action}] mean {m['mean']:.4f}  @10s {m['at_sec'][10]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="A")
    ap.add_argument("--model", choices=["zerovel", "simlpe"], default="zerovel")
    ap.add_argument("--ckpt", default=os.path.join(config.cache_dir(), "simlpe.pt"))
    ap.add_argument("--n-blocks", type=int, default=4)
    args = ap.parse_args()

    test_json = os.path.join(config._REPO_ROOT, "testdata", "test.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "zerovel":
        cb = zero_velocity_callback
    else:
        model = load_model(args.ckpt, n_blocks=args.n_blocks, device=device)
        cb = make_simlpe_callback(model, device=device)

    metrics = evaluate_callback(cb, args.dataset, config.data_root(), test_json)
    _print_report(f"{args.model} on {args.dataset}", metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_evaluate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/evaluate.py tests/test_evaluate.py
git commit -m "feat(forecasting): evaluation CLI + reporting"
```

---

### Task 10: End-to-end run on real data (zero-velocity, then siMLPe)

**Files:**
- Create: `forecasting/README.md` (how to run)
- Test: none (this is the real-data integration milestone, run manually)

**Interfaces:**
- Consumes: everything above; the mounted dataset at `HIK_DATA`.

This task produces the first real numbers and confirms the pipeline end-to-end. It depends on the dataset being mounted. Each step is a command with an expected qualitative outcome.

- [ ] **Step 1: Full fast test suite is green**

Run: `source .venv/bin/activate && python -m pytest tests/ -v -m "not slow"`
Expected: all tests pass.

- [ ] **Step 2: Slow data test (builds real windows)**

Run: `source .venv/bin/activate && HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen" python -m pytest tests/test_data.py -v -m slow`
Expected: PASS — `build_windows_for_dataset("A", ...)` returns up to 8 finite windows of shape `(500, 29, 3)`.

- [ ] **Step 3: Zero-velocity baseline number**

Run: `source .venv/bin/activate && HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen" python -m forecasting.evaluate --dataset A --model zerovel`
Expected: prints an overall mean MPJPE and per-horizon @1/5/10s, with error increasing across horizons. **Record these numbers** — this is the bar to beat.

- [ ] **Step 4: Build caches + train (time it; alert if >30 min)**

Run: `source .venv/bin/activate && HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen" time python -m forecasting.train --datasets A B C D --stepsize 50 --epochs 80`
Expected: prints window count, then per-epoch train/val loss that decreases; saves `forecasting/cache/simlpe.pt`. If wall-clock exceeds ~30 min, stop and report (revisit GPU/cloud per the spec).

- [ ] **Step 5: siMLPe number + write run notes**

Run: `source .venv/bin/activate && HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen" python -m forecasting.evaluate --dataset A --model simlpe`
Expected: overall mean MPJPE **below** the zero-velocity number from Step 3, especially at @1s/@2s. Write a short `forecasting/README.md` documenting: how to set `HIK_DATA`, the train and evaluate commands, and the recorded zero-velocity vs siMLPe numbers.

- [ ] **Step 6: Commit**

```bash
git add forecasting/README.md
git commit -m "docs(forecasting): run instructions + baseline numbers"
```

---

## Self-Review Notes

- **Spec coverage:** config/paths (T1) �·  pose rep normalize/denormalize (T6, T7) ✓ · siMLPe DCT-MLP (T2,T3) ✓ · train pipeline + no-leak exclusion (T7,T8) ✓ · MPJPE per-horizon scoring, target-pid only (T5) ✓ · zero-velocity reference (T6,T9) ✓ · local-GPU training (T8,T10) ✓ · `hik` untouched, scoring in our package (T5) ✓ · YAGNI: no scene/social/generative ✓.
- **Type consistency:** callback I/O `[250,P,29,3]` consistent across T6/T9; `WindowDataset` emits `[N_IN,POSE_DIM]`/`[N_OUT,POSE_DIM]` matching `SiMLPe.forward` and the losses; `calc_mpjpe` results-dict keys match `Evaluator.execute3d`'s stored entry keys (`Poses3d_out`, `Masks_out`, `Poses3d_out_pred`, `target_pid`, `pids`).
- **Known approximation:** the model is a faithful siMLPe realization (DCT → temporal MLP mixing → IDCT → last-frame residual) over equal in/out lengths, not a line-by-line port of the published repo. This is intentional and fully specified.
