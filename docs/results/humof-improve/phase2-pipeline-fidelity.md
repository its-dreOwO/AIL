# Phase 2 — pipeline fidelity: does the rebuilt pipeline still reproduce 109.71?

**Date:** 2026-07-15, completed 2026-07-16 · **Status:** ANSWERED on both sides — the rebuild is faithful

## Final answer (training side, 2026-07-16)

The GCP T4 baseline re-run finished all 70 epochs: **identical architecture,
identical data, rebuilt pipeline, same GPU class as the original.**

| | Original baseline | Rebuilt baseline | Δ |
|---|---:|---:|---:|
| Final epoch (69) path | 109.794 | 109.517 | −0.277 |
| Final epoch (69) pose | 66.796 | 66.853 | +0.057 |
| Late-epoch path (51–63, n=7) | 110.290 ± 0.853 | 110.150 ± 1.197 | −0.141 (t=−0.25) |
| Late-epoch pose (51–63, n=7) | 67.070 ± 0.253 | 67.212 ± 0.323 | +0.142 (t=+0.92) |

**Statistically indistinguishable.** Combined with the eval-side result below and
the exact filter-counter matches on both splits, every layer of the rebuild is
now verified: same data, same architecture, same eval behaviour, same training
outcome. The −0.14 mm rebuild delta doubles as the **measured training-side
reproduction noise** — the yardstick against which the gated arm's −0.34/−0.48 mm
deltas must be read (see [phase2-gated-ffn.md](phase2-gated-ffn.md)).

## Why this check exists

The VM that produced the original `70.pth` was deleted with no snapshot, taking the
filters, the preprocess script, and the env with it. Comparing a newly-trained
gated arm to the recorded **109.71** therefore changes five things at once: the
FFN, the GPU (T4→A100), Python (3.9→3.10), the `hik` package source, and the
**filter reconstruction**.

Two verifications already existed, but both are *static*: the filters reproduce
Phase 0's counts (13,162 admitted), and the param count matches exactly
(`10.01099M` baseline / `10.01377M` gated). They prove the **data subset** and the
**architecture** match. Neither proves the pipeline produces the same **number** —
a different claim that nothing tested.

`modal_app.py::eval_ckpt` tests it directly: run the surviving `70.pth` through
the rebuilt eval path and compare.

## Result

The eval was run **twice, identically** (same command, checkpoint, seed) — which
turned out to matter.

| Config | Path mean (mm) | Pose mean (mm) |
|---|---:|---:|
| Phase 1 deterministic baseline (T4, original pipeline) | 109.712 | 66.806 |
| evalref run 1: same `70.pth`, rebuilt pipeline (A100) | 109.756 | 66.792 |
| evalref run 2: **identical command** | 109.727 | 66.799 |
| **run 1 − run 2 (pure run-to-run noise)** | **0.028** | **0.008** |
| rebuilt mean (109.741) − Phase 1 baseline | **+0.030** | **−0.007** |

Independently re-confirmed inside both runs: `len(dataset)=13162` (filters) and
`total params: 10.01099M` — byte-identical to the original run's logged param
count, so `HUMOF_GATED_FFN=0` reproduces the original architecture exactly.

## Interpretation

**The rebuilt eval path is faithful.** Its mean sits **+0.030 mm** from Phase 1's
109.712 — *smaller than the pipeline's own run-to-run spread of 0.028 mm*. In
other words the rebuilt pipeline reproduces the original number to within
measurement precision. **109.71 remains a valid reference**, and the eval-side
contribution to the five-way confound is **~0.03 mm** on path, roughly an order of
magnitude below any plausible gated-FFN effect.

### The eval is NOT bit-reproducible — correcting a Phase 1 claim

Phase 1 states its runs are "bit-reproducible" after the `worker_init_fn` fix.
**Our rebuild is not**: two identical invocations differ by 0.028 mm on path.
That claim was evidently never tested by an actual repeat, and it does not hold
here.

Seeding is genuinely live — code inspection settles that much: `main.py:138`
defines `train(epoch, TRAIN, dataset)` as a *single* function used for both
training and eval, and `_seed_worker` (`main.py:181`) closes over `epoch` and is
passed as `worker_init_fn` to the one DataLoader (`main.py:197`). So the residual
noise is **not** the scene-point draw. The likely source is GPU nondeterminism in
the forward pass: pvcnn's voxelization uses `atomicAdd` scatters
(`pvcnn/modules/functional/src/voxelization/vox.cu`), whose float accumulation
order varies run to run.

A superseded hypothesis, recorded because it was nearly convincing: Phase 1's
pre-fix baseline was **109.76** and run 1 landed at **109.756**, which reads like
"the fix is inactive in the rebuild." Combined with the fact that Phase 1's exact
seeding formula was lost with the VM, the tidy story was "seeding is live but
draws different points, hence the offset." The repeat kills it — a seed-draw
offset would be *constant* across identical runs, and this one is not. The
near-match to 109.76 is coincidence inside a 0.03 mm band.

This does **not** contaminate the Phase 2 A/B (both arms run byte-identical code),
and it does **not** threaten Phase 1's headline: TTA4's −2.57 mm is ~90× this
noise, so it remains solidly real. What it does mean is that **any eval-side delta
below ~0.03 mm is unresolvable**, and n=2 gives only a crude spread, not a
standard deviation.

## The train-side filters are verified too (2026-07-15)

The design spec claimed the **train-side** filter reconstruction was unverifiable:
Phase 0 only recorded counts for D/test, so `primary_filterC` at threshold 0.6 on
A/B/C had nothing to check against. **That claim was wrong**, and the evidence was
already on disk: the original run's 55 MB `train_hik.log` was copied off the VM
before it was deleted, and it logs the train-side counters.

| Counter | Original (`train_hik.log`) | T4 rebuild | |
|---|---:|---:|:--|
| `__ct_filtered_1` | 482,397 | 482,397 | ✅ |
| `__ct_filtered_2` | 1,918 | 1,918 | ✅ |
| total sub sequences | 34,420 | 34,420 | ✅ |

Three independent counters matching exactly means the reconstruction reproduces
the original's **behaviour** on the training split, not merely its ballpark. Both
halves of the filter reconstruction — train (0.6 on A/B/C) and test (0.4 on D) —
are now verified against recorded ground truth, so `70.pth` was **not** trained on
a different dataset than the rebuild trains on. This removes the single largest
item from the five-way confound and is the strongest justification available for
comparing the gated arm to 109.71.

## What this does NOT establish

`eval_ckpt` reuses the **original weights**, so it isolates the eval half only.

It also does not bound **training-side nondeterminism**, which is the noise source
that actually matters for judging the gated delta: pvcnn's voxelization uses
`atomicAdd` scatters (`pvcnn/modules/functional/src/voxelization/vox.cu`) that
compound chaotically over 70 epochs, plausibly to the same order as the expected
effect. **Only the GCP T4 baseline re-run can bound that.** Do not quote the
±0.04 mm eval figure as "the noise floor" for the gated comparison — it is the
eval-side floor only, and the training-side floor is expected to be larger.
