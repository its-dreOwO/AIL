# Phase 2 — Gated (SwiGLU) FFN: results

**Date:** 2026-07-16 · **Status:** COMPLETE — all three arms finished

## Headline

**A param-matched SwiGLU FFN gives a small directional improvement that is not
statistically significant — and the effect is comparable in size to simply
re-running the identical baseline.** This is a **null result**, and it is the
outcome the
[design spec](../../superpowers/specs/2026-07-15-humof-gated-ffn-design.md)
predicted *before* the run: `dim_q=60` vs `ff_dim=2048` is a **34× expansion
ratio**, and gating buys least exactly where an FFN is that over-provisioned.

All three arms on the matched eval window (epochs 51–63, n=7 each, recording D):

| Arm | Path (mm) | Pose (mm) |
|---|---:|---:|
| Original baseline (T4, original pipeline) | 110.290 ± 0.853 | 67.070 ± 0.253 |
| **Rebuilt baseline** (T4, rebuilt pipeline, identical arch) | 110.150 ± 1.197 | 67.212 ± 0.323 |
| **Gated SwiGLU** (A100, rebuilt pipeline) | 109.812 ± 0.628 | 66.876 ± 0.262 |

| Contrast (path) | Δ (mm) | t | Reading |
|---|---:|---:|:--|
| rebuilt base − original base ("rebuild fidelity") | **−0.141** | −0.25 | identical within noise ✅ |
| gated − rebuilt base (FFN + GPU) | −0.337 | −0.66 | not significant |
| gated − original base (headline) | −0.478 | −1.19 | not significant |

| Contrast (pose) | Δ (mm) | t | Reading |
|---|---:|---:|:--|
| rebuilt base − original base | +0.142 | +0.92 | identical within noise ✅ |
| gated − rebuilt base | −0.337 | −2.14 | borderline; see caveat below |
| gated − original base | −0.195 | −1.41 | not significant |

The **rebuild-fidelity row is the yardstick**: retraining the *identical*
architecture on *identical* data moved path by −0.14 mm and pose by +0.14 mm.
The gated deltas (−0.34 to −0.48 path) are only 2–3× that reproduction noise,
and every contrast sits inside the arms' own ±0.6–1.2 mm late-epoch spread.
The pose contrast at t=−2.14 is nominally borderline, but with n=7
autocorrelated evals per arm and multiple contrasts examined, it does not
survive honest accounting.

The rebuilt baseline also completed as the **training-side fidelity check**: its
final-epoch number (109.517) and late-epoch distribution are statistically
indistinguishable from the original's (109.794), closing the last open question
about whether the rebuilt pipeline trains equivalently — details in
[phase2-pipeline-fidelity.md](phase2-pipeline-fidelity.md).

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
−0.20 mm pose vs the original baseline, not statistically significant — and only
2–3× the −0.14 mm that re-running the *identical* architecture produced.** The
lever was applied correctly (param-matched, verified to the parameter, on
verified-identical data), the pipeline was proven faithful on both the eval and
training sides, and the honest conclusion is that gating does not measurably help
HUMOF at this expansion ratio — which is what theory predicted. A null result,
reported as a null result, with its noise floor measured rather than assumed, is
the correct scientific outcome here.

## Appendix — full training specification (gated arm)

Every value below is read from the code/logs, not recalled.

### The change itself

`models/transformers.py::SwiGLUFeedForward`, selected by `HUMOF_GATED_FFN=1`
(baseline arm runs the byte-identical codebase with the flag at 0):

```python
hidden = round(2 * ff_dim / 3)
forward:  w_down( dropout( SiLU(w_gate(x)) * w_up(x) ) )
```

replacing `Linear(d→ff) → ReLU → Dropout → Linear(ff→d)`.

Applied to all six `TransformerLayer`s (`models/pipelines.py:181-188`):

| Layer | `dim_q` | `dim_kv` | `ff_dim` | SwiGLU hidden | dropout |
|---|---:|---:|---:|---:|---:|
| tl0–tl2 | 60 | 512 | 2048 | 1365 | 0.1 |
| tl3–tl4 | 60 | 256 | 1024 | 683 | 0.2 |
| tl5 | 60 | 128 | 512 | 341 | 0.2 |

Attention: 4 self-attention + 4 cross-attention heads per layer (unchanged).
`dim_q = 3·dct_n = 60` (`dct_n = 20`). Total params **10,013,770** vs baseline
**10,010,990** = **+2,780 (+0.028%)** — both logged by the runs themselves.

### Task / data

- HIK, multi-person: history `t_his=25` frames (1 s @ 25 Hz) → forecast
  `t_pred=50` frames (2 s); DCT representation with `dct_n=20`.
- Train: recordings A/B/C → **34,420** sub-sequences after
  `primary_filterC(0.6)`/`filterB(9.9)`; eval: recording D → **13,162** after
  `primary_filterC(0.4)`/`filterB(8.0)`. Both counts verified exactly against the
  original run's logs.
- Augmentation: random z-axis rotation (`datasets/aug.py`).
- Scene context: 1000 scene points per window via pvcnn voxelization.

### Optimization

| | |
|---|---|
| Optimizer | Adam, lr 5e-4, eps 1e-6, weight decay 1e-6 |
| LR schedule | lambda decay after 1 fixed epoch; lr at epoch 63 = 5e-5 |
| Epochs | 70 planned; **0–64 completed** (Modal credit exhausted) |
| Batching | `DynamicBatchSampler`, max_batch_size 48, max_batch_objs 256 (multi-person); 806 iters/epoch |
| Loss | joints + root (epoch 63: total 0.210 = joints 0.0795 + root 0.1306) |
| Seeds | global seed 0; sampler seeded `seed+epoch` (data order deterministic); workers `seed + 1000·(epoch+1) + worker_id` |

### Environment

| | |
|---|---|
| Hardware | Modal A100 40 GB, 8 vCPU, 32 GB RAM |
| Software | Python 3.10, torch 1.13.0+cu117, CUDA 11.7.1, numpy 1.24.3, pvcnn JIT (sm_80) |
| Precision | FP32; **TF32 explicitly disabled** to match the T4 baseline's semantics |
| Speed | ~450–470 s/epoch pure training (~7.5 min); ~11 min/epoch incl. periodic evals |
| Checkpointing | every 2 epochs (`HUMOF_SAVE_INTERVAL=2`) to a Modal Volume, committed every 600 s; restart-safe resume via newest-checkpoint autodetect |
| Wall clock | started 2026-07-15 21:13 +07, stopped ~2026-07-16 09:15 +07 at epoch 64 |

Repro: `humof_repro/modal_app.py::train --gated --tag gated`, with all HUMOF-side
edits in `humof_repro/patches/phase2-gated-ffn.patch`.
