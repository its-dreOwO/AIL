# Phase 1 — inference-time improvements (results)

**Date:** 2026-07-04 · HIK recording **D**, our reconstructed-filter subset (13,162 windows,
see [subset-report.md](subset-report.md)). All numbers are **relative A/Bs on the identical
subset** — absolute values are not paper-comparable, but the deltas are.

## Headline

**Test-time rotation averaging (TTA) is a clean, provable win: path −2.57 mm / pose −0.54 mm**
over the single-checkpoint baseline, with zero retraining. Checkpoint weight-averaging adds a
smaller −0.35 / −0.25 mm.

## Method

Two eval-only levers, no retraining:

1. **Checkpoint weight averaging** (`tools/avg_ckpt.py`): element-wise mean of the `model_dict`
   tensors of several epoch checkpoints → one model. Best window = **avg{50,60,70}** (ckpt 71).
   Wider windows (avg{30–70}) lower path further but start hurting pose (older checkpoints drag it).
2. **Test-time rotation averaging (TTA4)**: for each test sample, run the model on the input plus
   3 extra copies rotated about the z-axis (`[0, π/2, π, 3π/2]`), inverse-rotate each prediction
   back to the canonical frame, and average. The model already trains with z-rotation augmentation,
   so it is approximately rotation-equivariant; averaging reduces prediction variance. 4× eval cost.

**Determinism fix (important for a clean A/B).** The eval samples 1000 scene points per window via
`np.random.choice`, and the DataLoader workers did not seed numpy — giving ~0.03 mm run-to-run
noise. We added deterministic per-worker seeding (`worker_init_fn` seeding numpy/random/torch), so
every run below is **bit-reproducible**. (This shifts the absolute baseline by ~0.05 mm vs. the
pre-fix 109.76; all Phase-1 comparisons use the deterministic baseline.)

## Results (deterministic; recording D subset)

| Config | Path mean (mm) | Pose mean (mm) | Δ Path | Δ Pose |
|---|---:|---:|---:|---:|
| Baseline (ckpt 70) | 109.712 | 66.806 | — | — |
| Checkpoint avg{50,60,70} | 109.359 | 66.558 | −0.35 | −0.25 |
| **TTA4 (ckpt 70)** | **107.146** | 66.267 | **−2.57** | −0.54 |
| TTA4 + avg{50,60,70} | 107.393 | **66.228** | −2.32 | **−0.58** |

**Per-horizon** (baseline → TTA4, path mm): 0.5 s 51.30→50.36, 1 s 111.33→108.84,
2 s 223.38→218.04. TTA's benefit **grows with the horizon** (−5.3 mm at 2 s) — consistent with
averaging helping most where single-shot uncertainty is largest.

## Takeaways

- **Best path:** TTA4 on ckpt 70 (−2.57 mm, −2.3%). **Best pose:** TTA4 + avg (−0.58 mm).
- TTA and checkpoint-averaging **do not stack cleanly on path** (TTA4+avg path 107.39 is slightly
  worse than TTA4-alone 107.15) — TTA already captures most of the available gain; the smoother
  averaged weights help pose marginally but not path.
- All gains are **exactly reproducible** (deterministic eval, identical subset) — no seed/GPU
  confound. This is the provable, retraining-free deliverable; Phase 2 (a retrain) is optional upside.

## Reproduce
`~/HUMOF/run_phase1_det.sh` on the VM (sets `cp_iter` + `TTA_ANGLES`, runs `main.py`, saves each
`err_det_*.csv`). TTA hook + worker seeding live in `main.py` (backups `.bak_pretta`, `.bak_improve`).
