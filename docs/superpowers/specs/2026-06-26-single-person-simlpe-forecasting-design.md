# Single-person siMLPe forecasting baseline for HIK

**Date:** 2026-06-26
**Status:** Approved (design), pending implementation plan

## Purpose

Establish the first learned forecasting baseline on the *Humans in Kitchens* (HIK)
benchmark: predict 10 s of future 3D human motion from 10 s of past motion, for a
single person at a time, and plug it into the existing `Evaluator.execute3d`
protocol. This is **tier 1** of a longer roadmap (single-person → multi-person →
scene-aware → generative). It deliberately ignores other people and scene context;
those are later tiers.

Success criterion: a trained model that **clears the zero-velocity baseline** on
MPJPE, especially in the first few seconds of the horizon.

## Task framing

- **Horizon:** 250 observed frames → 250 predicted frames (10 s → 10 s @ 25 Hz).
- **Skeleton:** 29 joints × 3D = 87 dims per frame.
- **Per person, independently.** The eval callback receives everyone present and
  must return `[250, n_person, 29, 3]`; we run the model once per person and stack.
  Scoring focuses on the target pid.
- **Honest expectation:** siMLPe was designed for ~1 s horizons. At 10 s a
  deterministic model drifts toward a near-static / mean pose at far horizons.
  This is inherent, not a bug; the per-horizon metric will show it, and it
  motivates the later generative tier.

## Data

- **Root:** the mounted dataset at
  `/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen` (contains
  `poses/`, `scenes/{A..D}_scene/`, `body_models/SMPLX_NEUTRAL.npz`; the raw
  `190726/*.mp4` videos are ignored). The path has spaces, so it is supplied via
  the `HIK_DATA` env var / a config constant, never hardcoded with fragile quoting.
- **Scale:** 319 person-sequences across kitchens A–D, ~660k frames total
  (~440 min of recording).
- **Training windows:** `Scene.get_splits(length=500, stepsize=…)` across A–D,
  yielding 500-frame windows (250 in + 250 out). For each window, extract each
  person's sub-sequence, `backfill_masked` any gaps, and require validity across
  the full window. Estimated 20–40k single-person windows after filtering.
- **No train/test leak:** windows overlapping any frame referenced in
  `testdata/test.json` are dropped.
- **Caching:** preprocessed + normalized windows are cached to `.npz`. The one-time
  cost is `Scene` materialization (several minutes per kitchen); paid once.

## Pose representation

Full canonical normalization via the existing `hik.transforms.utils` helpers:

- `normalize3d(Poses3d, frame=249)` re-centers and re-orients each person's clip to
  a canonical facing (using left/right hip joints) at the **last observed frame**,
  returning per-person `Norm_params = (mu, R)`.
- The network predicts the future in this canonical space.
- `denormalize3d(pred, Norm_params)` inverts the transform back to world
  coordinates for scoring, using the **same** stored params. Round-trip
  invertibility is covered by the existing `test_normdenorm` unit test.

## Components — new `forecasting/` package (separate from the `hik` library)

The `hik` library API stays untouched; in particular we do **not** edit the
`hik/eval/mpjpe.py::calc_mpjpe` stub. Scoring lives in our package.

- **`config.py`** — `HIK_DATA` (defaults to the mounted path), `N_IN=250`,
  `N_OUT=250`, `N_JOINTS=29`, cache locations.
- **`dataset.py`** — build/cache training windows (per above) and expose a PyTorch
  `Dataset` over normalized windows.
- **`model.py`** — siMLPe: DCT-encode the 250-frame input (87 dims) → stack of MLP
  blocks (temporal + spatial mixing, LayerNorm) → IDCT → 250-frame output,
  predicted as a residual continuation in canonical space.
- **`train.py`** — Adam + cosine LR schedule; loss = L2 / MPJPE in normalized space
  plus a velocity (delta) term; train/val split by held-out windows; checkpoint the
  best model. Runs locally on the RTX 4060 (8 GB).
- **`evaluate.py`** —
  1. Implements the missing **`calc_mpjpe`** scoring (target pid only; reports
     overall MPJPE **and per-horizon** at 1 s/2 s/…/10 s).
  2. siMLPe eval callback: backfill → `normalize3d` → model → `denormalize3d` →
     stack into `[250, n_person, 29, 3]`.
  3. Trivial **zero-velocity callback** (repeat last observed frame) as the
     reference number — needed to confirm siMLPe is actually learning.

## Compute

- **Local RTX 4060 Laptop (8 GB), CUDA via torch 2.12.** Sufficient for siMLPe.
- **Estimated training time:** ~10–30 min (well under the 1 hr threshold) once
  windows are cached. Preprocessing is a one-time several-minute cost.
- **Cloud GPU (gcloud VPS) is deferred** to the later generative/diffusion tier,
  where model size and training time grow.

## Out of scope (YAGNI for this tier)

- Multi-person / social attention.
- Scene tokens / kitchen-object conditioning.
- Generative (diffusion / CVAE) output head.
- Editing the upstream `hik` library.

## Roadmap context

Tier 1 (this spec): single-person siMLPe. Then: tier 2 social attention, tier 3
scene-aware conditioning, tier 4 generative head for the long horizon. Each tier is
an independently runnable increment that produces a number.
