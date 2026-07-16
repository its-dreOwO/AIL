# Phase 2 — Gated (SwiGLU) FFN: results

**Date:** 2026-07-16 · **Status:** gated arm complete enough to report; baseline re-run still training

## Headline

**A param-matched SwiGLU FFN gives a small directional improvement that is not
statistically significant.** This is a **null result**, and it is the outcome the
[design spec](../../superpowers/specs/2026-07-15-humof-gated-ffn-design.md)
predicted *before* the run: `dim_q=60` vs `ff_dim=2048` is a **34× expansion
ratio**, and gating buys least exactly where an FFN is that over-provisioned.

| Metric (mm) | Baseline | Gated | Δ | t | |
|---|---:|---:|---:|---:|:--|
| Path | 110.290 ± 0.853 | **109.812** ± 0.628 | **−0.478** | −1.19 | not significant |
| Pose | 67.070 ± 0.253 | **66.876** ± 0.262 | **−0.195** | −1.41 | not significant |

Both arms, evals at epochs 51–63 (n=7 each), recording D.

## Why this is reported as a distribution, not two final numbers

The obvious comparison — gated@70 vs the recorded baseline 109.71 — would have
been **close to meaningless**, and the data says so plainly.

Eval error at a *single* epoch swings by **~±0.7 mm SD, ~2.1 mm range** across
consecutive late epochs, in **both** arms:

| Arm | Late evals (≥50) | Mean | SD | Range |
|---|---|---:|---:|---:|
| Baseline (original T4) | 10 | 110.094 | 0.769 | 2.132 |
| Gated (A100) | 7 | 109.812 | 0.628 | 2.139 |

That noise is **4–5× larger than the effect being measured**. The baseline's own
epoch-69 value (109.79) sits 0.30 mm *below* its late-epoch mean (110.09) — the
final epoch is not a special or "converged" number, just one draw from a noisy
distribution. Reporting `gated@70 − 109.79` would have let the stopping epoch
decide the headline.

Averaging over a **matched epoch range present in both arms (51–63)** is the
honest alternative: it uses the same epochs on both sides and averages down the
per-epoch noise. The near-identical spread between arms (SD 0.77 vs 0.63, range
2.13 vs 2.14) is also a useful consistency check — the rebuilt pipeline behaves
statistically like the original.

### Caveat that matters: n=7 is not 7 independent samples

The seven evals come from **one training run per arm**, so they are
autocorrelated points along a single trajectory, not independent draws. The
quoted `t` values therefore **overstate** confidence, and the true uncertainty is
larger than ±0.40 mm. Resolving a ~0.5 mm effect properly needs several
independent seeds per arm — which a ~9–24h-per-run budget cannot buy. **That
limitation is itself a finding**: with one run per arm, a sub-mm architectural
effect is not measurable, whatever the point estimate says.

## What the comparison controls, and what it does not

**Verified identical** (not assumed):

- **Architecture** — baseline logs `10.01099M` params, gated `10.01377M` = **+2,780
  exactly**, matching the 2/3-hidden param-matching prediction (+0.028%). Any delta
  is attributable to *gating*, not added capacity.
- **Data, both splits** — the reconstructed filters reproduce the original's
  counters exactly: train `__ct_filtered_1`=482,397 / `__ct_filtered_2`=1,918 /
  34,420 sub-sequences; test 13,162 admitted. See
  [phase2-pipeline-fidelity.md](phase2-pipeline-fidelity.md).
- **Eval pipeline** — the original `70.pth` re-scored through the rebuilt pipeline
  lands within its own run-to-run noise of the recorded value (109.741 vs 109.712).
- **Data order** — `DynamicBatchSampler` seeds a *local* generator with
  `seed + epoch`, so changing FFN parameter shapes does not shift which batches the
  model sees.
- **TF32** — pinned off, so the A100 matches T4 FP32 semantics.

**Not controlled:**

- **GPU** (gated on A100, baseline on T4) and **Python** (3.10 vs 3.9). The GCP T4
  baseline re-run (in progress) is what bounds this.
- **Training nondeterminism** — pvcnn's voxelization uses `atomicAdd` scatters
  (`pvcnn/.../vox.cu`) that compound chaotically over 70 epochs. Plausibly the same
  order as the effect, and not removable.
- **SwiGLU bundles two changes** — the multiplicative gate *and* ReLU→SiLU. A delta
  cannot be split between them; ReGLU would isolate the gate alone.

## Run notes

The gated arm trained **epochs 0–64 of 70** on a Modal A100 (~7.5 min/epoch pure
training, ~11 min/epoch including periodic auto-evals) and stopped when the Modal
credit was exhausted. **This does not affect the result**: the analysis uses the
matched 51–63 window, which was already complete, and epochs 64–70 would have
added at most three more autocorrelated points to a comparison that is
noise-limited rather than sample-limited.

Everything survived because checkpoints and `err.csv` were committed to a Modal
Volume, which is storage and outlives compute — the volume stayed fully readable
at zero credit. Final checkpoint `64.pth` and the eval history were pulled to
local disk.

## Bottom line for the report

The supervisor asked for the numbers, and the numbers are: **−0.48 mm path /
−0.20 mm pose, not statistically significant.** The lever was applied correctly
(param-matched, verified to the parameter, on verified-identical data), and the
honest conclusion is that it does not measurably help HUMOF at this expansion
ratio — which is what theory predicted. A null result, reported as a null result,
is the correct scientific outcome here.
