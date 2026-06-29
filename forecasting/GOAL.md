# Forecasting Goal Prompt — HIK Human-Motion Forecasting

> Single source of truth for the forecasting effort. Hand this to any agent/session
> to bring it fully up to speed. Last updated 2026-06-28.

## The task

`hik` ("Humans in Kitchens", NeurIPS 2023) is a **multi-person 3D human-motion
forecasting** benchmark with scene context. 4 kitchen recordings (A–D) @ 25 Hz.
We bring our own model and plug it into the vendored evaluation harness.

- **Protocol:** `Evaluator.execute3d(callback_fn, n_in=250, n_out=250)` — given 10 s
  of past motion for everyone present, predict the next 10 s.
- **Eval contract:** callback gets past poses/masks/kitchen/pids; in `[250,P,29,3]`
  → out `[250,P,29,3]` (a **single** deterministic prediction is scored).
- **Pose rep:** canonical `normalize3d` at frame 249 (last observed),
  `denormalize3d` for scoring.
- **Metric:** MPJPE via `mpjpe.mean_per_joint_l2_distance` (our scoring in
  `forecasting/metrics.py`). Lower is better. Reported overall + @1s/@5s/@10s.

## Current status (Tier 1 — single-person siMLPe baseline)

**Implemented, runs end-to-end, but still loses to the zero-velocity baseline.**
The full data → train → eval → MPJPE pipeline is **verified correct** — this is a
modeling problem, not a bug.

| horizon | zero-velocity (target to beat) | siMLPe iter 1 | siMLPe iter 2 | siMLPe iter 3 | scene-simlpe iter 4 |
|---|---|---|---|---|---|
| overall | **1.108** | 1.188 | 1.184 | 1.197 | 1.1845 |
| @1s | **0.520** | 0.607 | 0.572 | 0.537 | 0.5683 |
| @5s | **1.254** | 1.351 | 1.348 | 1.318 | 1.3411 |
| @10s | **1.422** | 1.472 | 1.462 | 1.707 | 1.4485 |

- **Iter 1** (A+B+C+D, stepsize 50, 80 epochs, 10,299 windows, Tesla T4): lost everywhere.
- **Iter 2** (horizon-weighted loss + lr 5e-4 + `--vel-weight 0.2` / `--horizon-floor 0.2`):
  improved every horizon (most at @1s) but **still loses everywhere**.
- **Iter 3** (velocity-output decoder + same weighted-loss config, 80 epochs, Tesla T4):
  improved @1s/@5s vs iter 2 but long-horizon drift got worse and **overall still loses**.
- **Iter 4** (nearest-object scene features + position decoder, 80 epochs, Tesla T4):
  roughly matched the weighted position decoder but **still loses overall**.

**Root cause — structural mean-collapse, not hyperparameters.** Predicting all 250
future frames jointly under position-space MPJPE rewards regressing to the mean
("repeat last frame + small noisy deltas"). Val loss flatlines by epoch ~5.
Loss-weighting tweaks have hit their ceiling.

**Key insight:** at a 10 s horizon, deterministic single-person motion is genuinely
multimodal/unpredictable in detail, so "freeze the last pose" (zero-velocity) is
near-optimal for the slow global trajectory. Beating it requires either a smarter
*output parameterization*, a *stochastic* model reported as best-of-N, or — the real
edge — *information zero-velocity does not have* (other people, scene/goal context).

## Models / methods to try (in priority order)

### A. Deterministic, siMLPe family (cheap, limited upside — same mean-collapse ceiling)
1. **Velocity / residual output reparameterization** *(next, highest-leverage cheap tweak)*
   — predict deltas from the last observed frame and/or a shorter autoregressive
   horizon instead of absolute 250-frame positions. Touches `model.py` + `callbacks.py`.
2. **History Repeats Itself (motion attention, Mao et al.)** — retrieves similar past
   sub-sequences and copies them forward; best of this group for periodic/cyclic
   kitchen motion (walking gait), exactly where zero-velocity has nothing.
3. **STS-GCN / MSR-GCN / DMGNN** — graph convs over the skeleton; structure-aware but
   still deterministic regression. Marginal.

### B. Stochastic / generative (Tier 4 — the right tool for 10 s)
Predict a distribution of futures, sample K, report best-of-N (ADE/FDE-style).
- **HumanMAC (diffusion, masked completion)** — *preferred*; clean single-stage,
  conditions on observed frames, fits the in/out contract almost directly.
- **BeLFusion** — latent behavioral diffusion, SOTA-ish on diverse long-horizon.
- **MID / DLow / GSPS** — lighter CVAE/diffusion diversity baselines.
- ⚠️ Harness scores ONE output. A stochastic model only shines if we report best-of-N
  or mean-sample as a separate result; straight MPJPE on the mean sample collapses again.

### C. Context-conditioned (Tiers 2 & 3 — the likely real win)
Use signal zero-velocity lacks. Single-person motion is unpredictable at 10 s, but
**goal-conditioned** motion is not: someone walking toward the sink reaches the sink.
- **Scene/goal-conditioned forecasting** — condition on kitchen object positions
  (`Kitchen` / `EnvironmentObject.is_inside`), predict the target object, anchor the
  trajectory to it.
- **Social attention (multi-person)** — condition each person on the others present.

## Current plan

1. **Stop grinding Tier 1.** Velocity-output siMLPe has now been trained/evaluated
   and still loses to zero-velocity overall.
2. **Nearest-object scene features are not enough.** Move to a stronger
   **scene/goal-conditioned** method with explicit goal inference/trajectory
   anchoring, or **HumanMAC** for a stochastic best-of-N result.
3. Keep all training on the GCP T4 VM; stop it after each run. Never train locally.

## Expected / acceptable score

- **Hard bar (acceptable):** beat zero-velocity overall — **MPJPE overall < 1.108**,
  ideally winning at every horizon (@1s < 0.520, @5s < 1.254, @10s < 1.422).
- **Stretch:** clear margin at short horizon (@1s ≲ 0.45) where motion is most
  predictable, while staying ≤ zero-velocity at @10s.
- **Stochastic track:** report best-of-K (K≈50) ADE/FDE alongside single-sample MPJPE,
  since a single deterministic sample cannot fairly represent a generative model.
- Anything that does not beat zero-velocity overall is **not yet a result** — it's
  diagnostic only.

## Infra & guardrails

- **Train on:** GCP VM `hik-simlpe-train` (project `project-b9a4f950-85a4-48f0-9ee`,
  zone `asia-southeast1-b`, n1-standard-8 + Tesla T4, ~30 GB RAM). Start to train,
  **stop (TERMINATE) when done** to avoid GPU billing. `gcloud` authed as
  drenguyen288@gmail.com.
- **Never run dataset/training/eval locally** — 14 GB RAM, OOMs (exit 137), has
  crashed the PC. Local = git, edits, reading, synthetic-array unit tests only.
- `build_or_load_windows` may OOM even on the 30 GB VM — apply a streaming/memmap fix
  before a full 4-dataset run.
- Runbook: `forecasting/README.md`. Specs/plans under `docs/superpowers/`.
