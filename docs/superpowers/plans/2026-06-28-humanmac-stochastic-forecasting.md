# HumanMAC Stochastic Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HumanMAC diffusion forecasting model and a best-of-K evaluation path so a stochastic model can be scored against (and ideally beat) the zero-velocity MPJPE bar.

**Architecture:** Whole-sequence DCT-domain DDPM with a transformer denoiser; observation is injected at inference by masked completion (RePaint-style imputation), no condition encoder. Evaluation runs one harness enumeration pass to capture inputs, batch-samples K futures per case, and scores with the existing `calc_best_of_k_mpjpe`.

**Tech Stack:** Python, NumPy, PyTorch, pytest/unittest, vendored HIK `Evaluator`, GCP T4 VM `hik-simlpe-train`.

## Global Constraints

- Pose rep: `N_IN = N_OUT = 250`, `WINDOW = 500`, `POSE_DIM = 87`, `FPS = 25` (from `forecasting/config.py`).
- Canonical normalization: `normalize3d(block, frame=N_IN-1)` for input, `denormalize3d` for scoring.
- Reuse the existing pose-window cache `forecasting/cache/windows_ABCD_s50.npy` (full 500-frame normalized windows). Do NOT build a new cache.
- best-of-K is an oracle metric: ALWAYS report best-of-K(K) AND single-sample MPJPE; label best-of-K as oracle. Print any K / L / ddim_steps / epoch values actually used (no silent caps).
- Do not modify vendored `hik/`, `testdata/`, `documentation/`, `notebooks/`.
- Local runs are synthetic/unit only (14 GB RAM OOMs on real data). Real train/eval on the T4 VM, stopped after.
- Local venv: `/opt/study/.venv/bin/python`. Remote: system `python3`.
- Test selection marker: non-data tests run under `-m "not slow"`.

---

## File Structure

- Create `forecasting/diffusion.py`: cosine β schedule, `GaussianDiffusion` (`q_sample`, `p_losses`), `ddim_sample` with masking hook. Pure diffusion math; no DCT, no model internals.
- Create `forecasting/humanmac.py`: `sinusoidal_embedding`, `HumanMACDenoiser` (transformer), `HumanMAC` wrapper (DCT/IDCT buffers, `forward` loss, `sample`).
- Create `forecasting/eval_bok.py`: capture-then-sample best-of-K evaluation + reporting.
- Modify `forecasting/data.py`: add `WholeWindowDataset`.
- Modify `forecasting/train.py`: add `train_humanmac` + `--model humanmac` CLI branch.
- Modify `forecasting/evaluate.py`: add `--model humanmac` → `eval_bok`.
- Modify `forecasting/README.md`, `forecasting/GOAL.md`: commands + result slot.
- Tests: `tests/test_diffusion.py`, `tests/test_humanmac.py`, `tests/test_eval_bok.py` (new); append to `tests/test_data.py`, `tests/test_train.py`.

---

### Task 1: Diffusion schedule + q_sample + p_losses

**Files:**
- Create: `tests/test_diffusion.py`
- Create: `forecasting/diffusion.py`

**Interfaces:**
- Produces: `cosine_beta_schedule(timesteps, s=0.008) -> Tensor[timesteps]`; `GaussianDiffusion(timesteps=1000)` with attrs `.timesteps:int`, buffers `.betas`, `.alphas_cumprod`; methods `q_sample(x0, t, noise) -> Tensor`, `p_losses(denoiser, x0, t) -> Tensor` (scalar MSE).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diffusion.py`:

```python
import torch

from forecasting.diffusion import GaussianDiffusion, cosine_beta_schedule


def test_cosine_schedule_betas_in_range_and_alphacumprod_decreasing():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas > 0) and torch.all(betas <= 0.999)
    abar = torch.cumprod(1.0 - betas, dim=0)
    assert torch.all(abar[1:] <= abar[:-1] + 1e-6)
    assert abar[0] <= 1.0 and abar[-1] > 0.0


def test_q_sample_is_clean_at_t0_and_noisy_at_tmax():
    diff = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(4, 10, 87)
    noise = torch.randn_like(x0)
    t0 = torch.zeros(4, dtype=torch.long)
    xt0 = diff.q_sample(x0, t0, noise)
    assert (xt0 - x0).abs().mean() < 1e-2

    tmax = torch.full((4,), 999, dtype=torch.long)
    xtmax = diff.q_sample(x0, tmax, noise)
    assert (xtmax - x0).abs().mean() > (xt0 - x0).abs().mean()


def test_p_losses_returns_finite_scalar_with_grad():
    diff = GaussianDiffusion(timesteps=50)

    class Echo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(87, 87)

        def forward(self, xt, t):
            return self.w(xt)

    denoiser = Echo()
    x0 = torch.randn(3, 10, 87)
    t = torch.randint(0, 50, (3,))
    loss = diff.p_losses(denoiser, x0, t)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert denoiser.w.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_diffusion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.diffusion'`.

- [ ] **Step 3: Implement schedule and GaussianDiffusion**

Create `forecasting/diffusion.py`:

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).to(torch.float32)


def _gather(values, t, ndim):
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, t, noise):
        abar = _gather(self.alphas_cumprod, t, x0.ndim)
        return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise

    def p_losses(self, denoiser, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = denoiser(xt, t)
        return F.mse_loss(pred, noise)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_diffusion.py -q`
Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/diffusion.py tests/test_diffusion.py
git commit -m "feat(forecasting): diffusion schedule, q_sample, p_losses"
```

---

### Task 2: DDIM sampler with masking hook

**Files:**
- Modify: `tests/test_diffusion.py`
- Modify: `forecasting/diffusion.py`

**Interfaces:**
- Consumes: `GaussianDiffusion` from Task 1.
- Produces: `ddim_sample(denoiser, diffusion, shape, obs_full_time, mask_time, dct_fn, idct_fn, ddim_steps, device="cpu") -> Tensor[B, T, C]`. `obs_full_time` is `[B, T, C]` with observed frames filled (future arbitrary); `mask_time` is `[T, 1]` (1.0 observed, 0.0 future); `dct_fn`/`idct_fn` map `[B,T,C]<->[B,L,C]`. The observed region of the returned trajectory equals `obs_full_time`'s observed region.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_diffusion.py`:

```python
from forecasting.diffusion import ddim_sample


def test_ddim_sample_pins_observed_region_with_full_dct():
    # Full (non-truncated) DCT so dct->idct is identity and masking is exact.
    T, C, L = 8, 3, 8
    dct_m = torch.linalg.qr(torch.randn(T, T))[0]  # orthonormal TxT
    idct_m = dct_m.t()

    def dct_fn(x):  # [B,T,C] -> [B,L,C]
        return torch.einsum("lt,btc->blc", dct_m[:L], x)

    def idct_fn(X):  # [B,L,C] -> [B,T,C]
        return torch.einsum("tl,blc->btc", idct_m[:, :L], X)

    diff = GaussianDiffusion(timesteps=20)
    denoiser = lambda xt, t: torch.zeros_like(xt)

    n_in = 4
    obs = torch.randn(2, T, C)
    obs[:, n_in:] = 0.0
    mask = torch.zeros(T, 1)
    mask[:n_in] = 1.0

    out = ddim_sample(
        denoiser, diff, (2, L, C), obs, mask, dct_fn, idct_fn, ddim_steps=5
    )
    assert out.shape == (2, T, C)
    assert torch.allclose(out[:, :n_in], obs[:, :n_in], atol=1e-4)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_diffusion.py::test_ddim_sample_pins_observed_region_with_full_dct -q`
Expected: FAIL with import error for `ddim_sample`.

- [ ] **Step 3: Implement ddim_sample**

Append to `forecasting/diffusion.py`:

```python
@torch.no_grad()
def ddim_sample(
    denoiser, diffusion, shape, obs_full_time, mask_time, dct_fn, idct_fn,
    ddim_steps, device="cpu",
):
    B = shape[0]
    abar = diffusion.alphas_cumprod.to(device)
    obs_full_time = obs_full_time.to(device)
    mask_time = mask_time.to(device)
    obs_latent = dct_fn(obs_full_time)

    X = torch.randn(shape, device=device)
    seq = torch.linspace(diffusion.timesteps - 1, 0, ddim_steps).round().long().tolist()
    seq_prev = seq[1:] + [-1]

    for t, t_prev in zip(seq, seq_prev):
        tb = torch.full((B,), t, device=device, dtype=torch.long)
        eps = denoiser(X, tb)
        a_t = abar[t]
        x0 = (X - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        a_prev = abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        X = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps

        # masked completion: pin observed region at noise level t_prev (clean at end)
        level = max(t_prev, 0)
        known_latent = (
            abar[level].sqrt() * obs_latent
            + (1.0 - abar[level]).sqrt() * torch.randn_like(obs_latent)
        )
        x_time = idct_fn(X)
        known_time = idct_fn(known_latent)
        x_time = mask_time * known_time + (1.0 - mask_time) * x_time
        X = dct_fn(x_time)

    out = idct_fn(X)
    out = mask_time * obs_full_time + (1.0 - mask_time) * out  # hard pin observed
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_diffusion.py -q`
Expected: PASS, `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/diffusion.py tests/test_diffusion.py
git commit -m "feat(forecasting): DDIM sampler with masked-completion hook"
```

---

### Task 3: HumanMAC transformer denoiser

**Files:**
- Create: `tests/test_humanmac.py`
- Create: `forecasting/humanmac.py`

**Interfaces:**
- Produces: `sinusoidal_embedding(t, dim) -> Tensor[len(t), dim]`; `HumanMACDenoiser(pose_dim=87, n_coeff=125, d_model=512, n_layers=8, n_heads=8, ff_mult=4)` with `forward(X[B,n_coeff,pose_dim], t[B]) -> Tensor[B,n_coeff,pose_dim]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_humanmac.py`:

```python
import torch

from forecasting.humanmac import HumanMACDenoiser, sinusoidal_embedding


def test_sinusoidal_embedding_shape():
    t = torch.arange(5)
    emb = sinusoidal_embedding(t, 16)
    assert emb.shape == (5, 16)
    assert torch.isfinite(emb).all()


def test_denoiser_forward_shape():
    model = HumanMACDenoiser(pose_dim=87, n_coeff=20, d_model=32, n_layers=2, n_heads=4)
    X = torch.randn(3, 20, 87)
    t = torch.randint(0, 1000, (3,))
    out = model(X, t)
    assert out.shape == (3, 20, 87)
    assert torch.isfinite(out).all()


def test_denoiser_output_depends_on_timestep():
    torch.manual_seed(0)
    model = HumanMACDenoiser(pose_dim=87, n_coeff=20, d_model=32, n_layers=2, n_heads=4)
    X = torch.randn(2, 20, 87)
    out_a = model(X, torch.zeros(2, dtype=torch.long))
    out_b = model(X, torch.full((2,), 999, dtype=torch.long))
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_humanmac.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.humanmac'`.

- [ ] **Step 3: Implement the denoiser**

Create `forecasting/humanmac.py`:

```python
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim):
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=-1)
    return emb


class HumanMACDenoiser(nn.Module):
    def __init__(self, pose_dim=87, n_coeff=125, d_model=512, n_layers=8, n_heads=8, ff_mult=4):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(pose_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_coeff, d_model))
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * ff_mult,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, pose_dim)

    def forward(self, X, t):
        temb = self.t_mlp(sinusoidal_embedding(t, self.d_model)).unsqueeze(1)
        h = self.in_proj(X) + self.pos + temb
        h = self.encoder(h)
        return self.out(h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_humanmac.py -q`
Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/humanmac.py tests/test_humanmac.py
git commit -m "feat(forecasting): HumanMAC transformer denoiser"
```

---

### Task 4: HumanMAC wrapper (DCT, training loss, sampling)

**Files:**
- Modify: `tests/test_humanmac.py`
- Modify: `forecasting/humanmac.py`

**Interfaces:**
- Consumes: `HumanMACDenoiser` (Task 3); `GaussianDiffusion`, `ddim_sample` (Tasks 1-2); `get_dct_matrix` (`forecasting/dct.py`).
- Produces: `HumanMAC(pose_dim=87, window=500, n_in=250, n_coeff=125, d_model=512, n_layers=8, n_heads=8, timesteps=1000, ddim_steps=50)` with `forward(x0_time[B,window,pose_dim]) -> scalar loss`, and `sample(obs_time[n_in,pose_dim], k, ddim_steps=None) -> Tensor[k, n_out, pose_dim]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_humanmac.py`:

```python
from forecasting.humanmac import HumanMAC


def _tiny_humanmac():
    return HumanMAC(
        pose_dim=87, window=8, n_in=4, n_coeff=8,
        d_model=16, n_layers=1, n_heads=2, timesteps=20, ddim_steps=4,
    )


def test_humanmac_forward_is_finite_scalar_loss():
    model = _tiny_humanmac()
    x0 = torch.randn(2, 8, 87)
    loss = model(x0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_humanmac_sample_shape_and_stochastic():
    torch.manual_seed(0)
    model = _tiny_humanmac()
    obs = torch.randn(4, 87)  # n_in=4
    a = model.sample(obs, k=3)
    b = model.sample(obs, k=3)
    assert a.shape == (3, 4, 87)  # k, n_out=4, pose_dim
    assert torch.isfinite(a).all()
    assert not torch.allclose(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_humanmac.py::test_humanmac_forward_is_finite_scalar_loss tests/test_humanmac.py::test_humanmac_sample_shape_and_stochastic -q`
Expected: FAIL with import error for `HumanMAC`.

- [ ] **Step 3: Implement the wrapper**

Append to `forecasting/humanmac.py` (add imports at top of file too):

```python
# add near the top imports:
from forecasting.dct import get_dct_matrix
from forecasting.diffusion import GaussianDiffusion, ddim_sample


class HumanMAC(nn.Module):
    def __init__(
        self, pose_dim=87, window=500, n_in=250, n_coeff=125,
        d_model=512, n_layers=8, n_heads=8, timesteps=1000, ddim_steps=50,
    ):
        super().__init__()
        self.window = window
        self.n_in = n_in
        self.n_out = window - n_in
        self.n_coeff = n_coeff
        self.ddim_steps = ddim_steps

        dct_m, idct_m = get_dct_matrix(window)
        self.register_buffer("dct_l", torch.tensor(dct_m[:n_coeff], dtype=torch.float32))
        self.register_buffer("idct_l", torch.tensor(idct_m[:, :n_coeff], dtype=torch.float32))

        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.denoiser = HumanMACDenoiser(
            pose_dim=pose_dim, n_coeff=n_coeff, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads,
        )

    def dct(self, x_time):
        return torch.einsum("lt,btc->blc", self.dct_l, x_time)

    def idct(self, X):
        return torch.einsum("tl,blc->btc", self.idct_l, X)

    def forward(self, x0_time):
        X0 = self.dct(x0_time)
        t = torch.randint(0, self.diffusion.timesteps, (X0.shape[0],), device=X0.device)
        return self.diffusion.p_losses(self.denoiser, X0, t)

    @torch.no_grad()
    def sample(self, obs_time, k, ddim_steps=None):
        device = self.dct_l.device
        steps = ddim_steps or self.ddim_steps
        pose_dim = obs_time.shape[-1]
        obs_full = torch.zeros(k, self.window, pose_dim, device=device)
        obs_full[:, : self.n_in] = obs_time.to(device)[None]
        mask = torch.zeros(self.window, 1, device=device)
        mask[: self.n_in] = 1.0
        traj = ddim_sample(
            self.denoiser, self.diffusion, (k, self.n_coeff, pose_dim),
            obs_full, mask, self.dct, self.idct, steps, device=device,
        )
        return traj[:, self.n_in :]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_humanmac.py -q`
Expected: PASS, `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/humanmac.py tests/test_humanmac.py
git commit -m "feat(forecasting): HumanMAC wrapper with DCT, loss, sampling"
```

---

### Task 5: WholeWindowDataset

**Files:**
- Modify: `tests/test_data.py`
- Modify: `forecasting/data.py`

**Interfaces:**
- Produces: `WholeWindowDataset(windows[N,WINDOW,29,3])` with `__len__` and `__getitem__(i) -> Tensor[WINDOW, POSE_DIM] float32`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data.py`:

```python
from forecasting.data import WholeWindowDataset


def test_whole_window_dataset_returns_full_window_x0():
    windows = np.random.default_rng(3).standard_normal(
        (4, config.WINDOW, 29, 3)
    ).astype(np.float32)
    ds = WholeWindowDataset(windows)
    assert len(ds) == 4
    x0 = ds[2]
    assert x0.shape == (config.WINDOW, config.POSE_DIM)
    assert torch.is_tensor(x0) and x0.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_data.py::test_whole_window_dataset_returns_full_window_x0 -q`
Expected: FAIL with import error for `WholeWindowDataset`.

- [ ] **Step 3: Implement the dataset**

Append to `forecasting/data.py` (below `WindowDataset`):

```python
class WholeWindowDataset(torch.utils.data.Dataset):
    """Yields the full normalized window as a diffusion target x0."""

    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        window = self.windows[i].reshape(WINDOW, POSE_DIM)
        return torch.tensor(window, dtype=torch.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_data.py -q -m "not slow"`
Expected: PASS for non-slow data tests.

- [ ] **Step 5: Commit**

```bash
git add forecasting/data.py tests/test_data.py
git commit -m "feat(forecasting): whole-window dataset for diffusion training"
```

---

### Task 6: HumanMAC training path

**Files:**
- Modify: `tests/test_train.py`
- Modify: `forecasting/train.py`

**Interfaces:**
- Consumes: `HumanMAC` (Task 4), `WholeWindowDataset` (Task 5), `build_or_load_windows` (existing).
- Produces: `train_humanmac(windows, *, epochs, batch_size=64, lr=2e-4, val_frac=0.05, device=None, n_coeff=125, d_model=512, n_layers=8, n_heads=8, timesteps=1000, ddim_steps=50, seed=0, out_path="__default__") -> dict` with keys `train_loss`, `val_loss`, `best_val`, `ckpt`; default checkpoint `humanmac.pt`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
from forecasting.train import train_humanmac


def test_train_humanmac_runs_one_epoch_and_writes_ckpt(tmp_path):
    windows = np.random.default_rng(0).standard_normal(
        (6, config.WINDOW, 29, 3)
    ).astype(np.float32)
    out = tmp_path / "humanmac.pt"
    hist = train_humanmac(
        windows, epochs=1, batch_size=3, device="cpu",
        n_coeff=8, d_model=16, n_layers=1, n_heads=2,
        timesteps=20, ddim_steps=2, out_path=str(out),
    )
    assert len(hist["train_loss"]) == 1
    assert np.isfinite(hist["train_loss"][0])
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_train.py::test_train_humanmac_runs_one_epoch_and_writes_ckpt -q`
Expected: FAIL with import error for `train_humanmac`.

- [ ] **Step 3: Implement train_humanmac and CLI branch**

Add imports to `forecasting/train.py`:

```python
from forecasting.data import (
    SceneWindowDataset,
    WholeWindowDataset,
    WindowDataset,
    build_or_load_scene_windows,
    build_or_load_windows,
)
from forecasting.humanmac import HumanMAC
```

Add `train_humanmac` (below `train`):

```python
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
```

Update CLI `--model` choices and `main()`:

```python
parser.add_argument(
    "--model", choices=["simlpe", "scene-simlpe", "humanmac"], default="simlpe"
)
parser.add_argument("--n-coeff", type=int, default=125)
parser.add_argument("--d-model", type=int, default=512)
parser.add_argument("--n-layers", type=int, default=8)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--ddim-steps", type=int, default=50)
```

In `main()`, dispatch humanmac before the existing branches:

```python
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
```

(Keep the existing scene/simlpe path below this guard.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_train.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add forecasting/train.py tests/test_train.py
git commit -m "feat(forecasting): HumanMAC training path"
```

---

### Task 7: best-of-K evaluation (capture-then-sample)

**Files:**
- Create: `tests/test_eval_bok.py`
- Create: `forecasting/eval_bok.py`

**Interfaces:**
- Consumes: `calc_best_of_k_mpjpe`, `calc_mpjpe` (`forecasting/metrics.py`); `zero_velocity_callback`, `backfill_masked`, `normalize3d`, `denormalize3d`.
- Produces:
  - `capture_inputs(evaluator) -> list[dict]` (one input dict per test case, each carries `action` plus the harness keys incl. `Poses3d_out`, `Masks_out`, `pids`, but here we also stash `target_pid`).
  - `build_bok_results(cases, sample_case, k) -> (results_bok, results_single)` where `sample_case(inp) -> ndarray[k, n_out, n_person, 29, 3]`.
  - `evaluate_humanmac_bok(evaluator, sample_case, k) -> (bok_metrics, single_metrics)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_bok.py`:

```python
import numpy as np

from forecasting.eval_bok import build_bok_results, capture_inputs


class FakeEvaluator:
    def __init__(self, cases):
        self._cases = cases

    def execute3d(self, callback_fn, **kwargs):
        results = {}
        for c in self._cases:
            results.setdefault(c["action"], []).append({})
            callback_fn(c)
        return results


def _case(action, target_pid, pids):
    T, P = 250, len(pids)
    return {
        "Poses3d_in": np.zeros((250, P, 29, 3), dtype=np.float32),
        "Masks_in": np.ones((250, P), dtype=np.float32),
        "Poses3d_out": np.zeros((T, P, 29, 3), dtype=np.float32),
        "Masks_out": np.ones((T, P), dtype=np.float32),
        "n_out": 250,
        "pids": pids,
        "frames_in": list(range(250)),
        "action": action,
        "pid": target_pid,
    }


def test_capture_inputs_collects_all_cases():
    cases = [_case("walking", 7, [7, 9]), _case("sink", 3, [3])]
    captured = capture_inputs(FakeEvaluator(cases))
    assert len(captured) == 2
    assert captured[0]["target_pid"] == 7
    assert captured[1]["action"] == "sink"


def test_build_bok_results_assembles_samples_and_scores():
    cases = [_case("walking", 7, [7])]
    captured = capture_inputs(FakeEvaluator(cases))

    def sample_case(inp):
        k, n_out, P = 2, inp["n_out"], len(inp["pids"])
        out = np.zeros((k, n_out, P, 29, 3), dtype=np.float32)
        out[0] += 1.0  # bad sample
        out[1] += 0.1  # good sample
        return out

    results_bok, results_single = build_bok_results(captured, sample_case, k=2)
    entry = results_bok["walking"][0]
    assert entry["Poses3d_out_pred_samples"].shape == (2, 250, 1, 29, 3)
    assert results_single["walking"][0]["Poses3d_out_pred"].shape == (250, 1, 29, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval_bok.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.eval_bok'`.

- [ ] **Step 3: Implement eval_bok core**

Create `forecasting/eval_bok.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval_bok.py -q`
Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/eval_bok.py tests/test_eval_bok.py
git commit -m "feat(forecasting): best-of-K capture-then-sample evaluation"
```

---

### Task 8: HumanMAC sampler-callback + evaluate wiring

**Files:**
- Modify: `tests/test_eval_bok.py`
- Modify: `forecasting/eval_bok.py`
- Modify: `forecasting/evaluate.py`

**Interfaces:**
- Consumes: `HumanMAC` (Task 4), `build_bok_results`/`evaluate_humanmac_bok` (Task 7).
- Produces: `make_humanmac_sample_case(model, k, ddim_steps=None, device="cpu") -> (inp -> ndarray[k, n_out, P, 29, 3])`; `load_humanmac(ckpt, device, **model_kwargs) -> HumanMAC`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_bok.py`:

```python
import torch

from forecasting.eval_bok import make_humanmac_sample_case
from forecasting.humanmac import HumanMAC


def test_humanmac_sample_case_shapes_and_target_only():
    model = HumanMAC(
        pose_dim=87, window=500, n_in=250, n_coeff=8,
        d_model=16, n_layers=1, n_heads=2, timesteps=20, ddim_steps=2,
    )
    sample_case = make_humanmac_sample_case(model, k=2, device="cpu")
    inp = _case("walking", 7, [7, 9])
    out = sample_case(inp)
    assert out.shape == (2, 250, 2, 29, 3)
    assert np.isfinite(out).all()
    # non-target person (pid 9, index 1) is identical across samples (zero-velocity)
    assert np.allclose(out[0, :, 1], out[1, :, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval_bok.py::test_humanmac_sample_case_shapes_and_target_only -q`
Expected: FAIL with import error for `make_humanmac_sample_case`.

- [ ] **Step 3: Implement the sampler-callback**

Add imports + function to `forecasting/eval_bok.py`:

```python
import torch

from hik.transforms.utils import backfill_masked, denormalize3d, normalize3d
from forecasting.config import N_IN, POSE_DIM


def make_humanmac_sample_case(model, k, ddim_steps=None, device="cpu"):
    model = model.to(device)
    model.eval()

    def sample_case(inp):
        poses_in = np.copy(inp["Poses3d_in"])
        masks_in = np.copy(inp["Masks_in"])
        n_out = inp["n_out"]
        pids = inp["pids"]
        n_person = poses_in.shape[1]
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
```

- [ ] **Step 4: Wire evaluate.py**

Add to `forecasting/evaluate.py` imports:

```python
from forecasting.eval_bok import evaluate_humanmac_bok, make_humanmac_sample_case
from forecasting.humanmac import HumanMAC
```

Add a loader:

```python
def load_humanmac(ckpt, device="cpu", n_coeff=125, d_model=512, n_layers=8,
                  n_heads=8, timesteps=1000, ddim_steps=50):
    model = HumanMAC(
        pose_dim=config.POSE_DIM, window=config.WINDOW, n_in=config.N_IN,
        n_coeff=n_coeff, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        timesteps=timesteps, ddim_steps=ddim_steps,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model
```

Extend CLI choices and add args:

```python
parser.add_argument(
    "--model", choices=["zerovel", "simlpe", "scene-simlpe", "humanmac"], default="zerovel"
)
parser.add_argument("--k", type=int, default=50)
parser.add_argument("--n-coeff", type=int, default=125)
parser.add_argument("--d-model", type=int, default=512)
parser.add_argument("--n-layers", type=int, default=8)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--ddim-steps", type=int, default=50)
```

Add a humanmac branch in `main()` (before the zerovel/else dispatch), using the vendored Evaluator directly:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/study/.venv/bin/python -m pytest tests/test_eval_bok.py -q`
Expected: PASS, `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add forecasting/eval_bok.py forecasting/evaluate.py tests/test_eval_bok.py
git commit -m "feat(forecasting): HumanMAC best-of-K evaluate path"
```

---

### Task 9: Docs + full local verification

**Files:**
- Modify: `forecasting/README.md`
- Modify: `forecasting/GOAL.md`

- [ ] **Step 1: Add a HumanMAC section to `forecasting/README.md`**

```markdown
## Train + evaluate HumanMAC (stochastic, best-of-K)

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen"
python -m forecasting.train --model humanmac \
  --datasets A B C D --stepsize 50 --epochs 500 \
  --batch-size 64 --lr 2e-4 \
  --n-coeff 125 --d-model 512 --n-layers 8 --n-heads 8 \
  --timesteps 1000 --ddim-steps 50
# writes forecasting/cache/humanmac.pt (reuses windows_ABCD_s50.npy)

python -m forecasting.evaluate --dataset A --model humanmac \
  --ckpt forecasting/cache/humanmac.pt --k 50 \
  --n-coeff 125 --d-model 512 --n-layers 8 --n-heads 8 \
  --timesteps 1000 --ddim-steps 50
```

Reports best-of-K (oracle) and single-sample MPJPE. best-of-K uses ground
truth to pick the best of K samples and is NOT comparable to the deterministic
zero-velocity bar; both numbers are always printed.
```

- [ ] **Step 2: Update `forecasting/GOAL.md`** — add a HumanMAC row/section noting the stochastic best-of-K track is implemented and pending its first VM result.

- [ ] **Step 3: Run full local verification**

Run:
```bash
/opt/study/.venv/bin/python -m pytest tests/ -q -m "not slow"
/opt/study/.venv/bin/python -m unittest forecasting.test_losses
```
Expected: all non-slow tests pass; loss unittest OK.

- [ ] **Step 4: Commit**

```bash
git add forecasting/README.md forecasting/GOAL.md
git commit -m "docs(forecasting): document HumanMAC stochastic best-of-K track"
```

---

### Task 10: Remote VM training, evaluation, and result

**Files:**
- Modify after run: `forecasting/README.md`, `forecasting/GOAL.md`

- [ ] **Step 1: Verify VM status**

```bash
gcloud compute instances describe hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b \
  --format='value(status)'
```
Expected: `TERMINATED` or `RUNNING`.

- [ ] **Step 2: Start VM if TERMINATED**

```bash
gcloud compute instances start hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b
```

- [ ] **Step 3: Copy changed files to the VM**

```bash
gcloud compute scp \
  forecasting/diffusion.py forecasting/humanmac.py forecasting/eval_bok.py \
  forecasting/data.py forecasting/train.py forecasting/evaluate.py \
  forecasting/metrics.py forecasting/README.md forecasting/GOAL.md \
  hik-simlpe-train:~/hik/forecasting/ \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b

gcloud compute scp \
  tests/test_diffusion.py tests/test_humanmac.py tests/test_eval_bok.py \
  tests/test_data.py tests/test_train.py tests/test_metrics.py \
  hik-simlpe-train:~/hik/tests/ \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b
```

- [ ] **Step 4: Remote unit tests before training**

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b \
  --command='cd ~/hik && python3 -m pytest tests/ -q -m "not slow" && python3 -m unittest forecasting.test_losses'
```
Expected: all pass.

- [ ] **Step 5: Train (multi-hour; run detached with nohup)**

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && nohup python3 -u -m forecasting.train --model humanmac --datasets A B C D --stepsize 50 --epochs 500 --batch-size 64 --lr 2e-4 --n-coeff 125 --d-model 512 --n-layers 8 --n-heads 8 --timesteps 1000 --ddim-steps 50 > ~/humanmac_train.log 2>&1 &'
```
Poll `~/humanmac_train.log` until training completes and `forecasting/cache/humanmac.pt` exists. If wall-clock is excessive, reduce `--epochs` and note it in the result (no silent caps).

- [ ] **Step 6: Evaluate (all four datasets)**

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && for D in A B C D; do python3 -m forecasting.evaluate --dataset $D --model humanmac --ckpt forecasting/cache/humanmac.pt --k 50 --n-coeff 125 --d-model 512 --n-layers 8 --n-heads 8 --timesteps 1000 --ddim-steps 50; done'
```
Expected: best-of-50 + single-sample overall/@1s/@5s/@10s per dataset. If best-of-50 is close to but not under 1.108, re-run eval with `--k 100` (eval-only, no retrain) and record the K used.

- [ ] **Step 7: Stop VM and verify**

```bash
gcloud compute instances stop hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b --quiet
gcloud compute instances describe hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee --zone=asia-southeast1-b \
  --format='value(status)'
```
Expected: `TERMINATED`.

- [ ] **Step 8: Record result**

Update `forecasting/README.md` and `forecasting/GOAL.md` with a `Result (2026-06-28 HumanMAC best-of-K)` section:
- config: `humanmac`, K, n_coeff, d_model/layers/heads, timesteps, ddim_steps, epochs, datasets, Tesla T4.
- table columns: `horizon`, `zero-velocity`, `HumanMAC best-of-K [oracle]`, `HumanMAC single-sample`.
- rows: `overall`, `@1s`, `@5s`, `@10s`.
- conclusion: if best-of-K overall `< 1.108`, mark the stochastic track as clearing the bar (oracle); state the single-sample number honestly; if not, mark diagnostic and propose next levers (more epochs, larger K, more DCT coeffs / ddim steps, classifier-free guidance).

- [ ] **Step 9: Commit result docs**

```bash
git add forecasting/README.md forecasting/GOAL.md
git commit -m "docs(forecasting): record HumanMAC best-of-K result"
```

---

## Final Verification Checklist

- [ ] Local non-slow pytest passes (diffusion, humanmac, data, train, eval_bok, metrics).
- [ ] Local `forecasting.test_losses` unittest passes.
- [ ] `ddim_sample` masking-invariant test passes (observed region pinned).
- [ ] Remote non-slow pytest passes before training.
- [ ] Remote training writes `forecasting/cache/humanmac.pt`.
- [ ] Remote evaluation prints best-of-K (oracle) AND single-sample MPJPE; K/ddim_steps printed.
- [ ] VM verified `TERMINATED` after the run.
- [ ] `forecasting/README.md` and `forecasting/GOAL.md` record the result with the oracle caveat.
- [ ] If best-of-K overall `< 1.108`, the stochastic-track goal is met (oracle); otherwise the goal stays active with next levers listed.
