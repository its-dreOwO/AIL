# Phase 2 — Gated (SwiGLU) FFN in the HUMOF transformer layer

**Date:** 2026-07-15 · **Status:** approved, in execution · **Deadline:** Fri 2026-07-17

## Origin

Supervisor directive: replace the FFN in HUMOF's Transformer Layer with a gated
FFN and report the numbers. This is the Phase 2 architecture lever, chosen by the
supervisor rather than by the Phase 0/1-informed ranking in the
[2026-07-04 design](2026-07-04-humof-baseline-improvement-design.md) (whose
leading candidate was iterative multi-person refinement).

Secondary motivation from the user: Phase 1's wins (TTA4, checkpoint averaging)
are *inference-time* tricks that never touch the model. This lever changes the
pipeline itself.

## What we're changing

`models/transformers.py:23-28` — the FFN inside `TransformerLayer` is a plain
2-layer ReLU MLP:

```python
nn.Sequential(nn.Linear(dim_q, ff_dim), nn.ReLU(), nn.Dropout(p), nn.Linear(ff_dim, dim_q))
```

Replaced by a **parameter-matched SwiGLU**, behind the `GATED_FFN` env flag so
both arms run byte-identical code and differ only in that flag.

### Parameter matching (decided, not defaulted)

SwiGLU needs three projections instead of two. At equal `ff_dim` it would add
~50% FFN params, confounding *gating* with *added capacity*. We scale hidden by
2/3: `h = round(2*ff_dim/3)`.

| Layer | `ff_dim` | `h` | baseline params | SwiGLU params | Δ |
|---|---:|---:|---:|---:|---:|
| tl0-2 | 2048 | 1365 | 247,868 | 248,490 | +0.25% |
| tl3-4 | 1024 | 683 | 123,964 | 124,366 | +0.32% |
| tl5 | 512 | 341 | 62,012 | 62,122 | +0.18% |
| **total** | | | **1,053,544** | **1,056,324** | **+0.264%** |

Verified numerically. Any delta is therefore attributable to gating, not capacity.

### Known limitation of the chosen variant

SwiGLU bundles **two** changes vs. the baseline: the multiplicative gate *and*
ReLU→SiLU. A delta cannot be split between them. ReGLU would have isolated the
gate alone; SwiGLU was chosen as the standard, best-performing variant and what
"gated FFN" normally means. This must be stated in the write-up.

## Prior expectation (state it up front, not after seeing the result)

**We expect this to be a small effect, possibly unmeasurable.** `dim_q = 3*dct_n
= 60` while `ff_dim = 2048` — a **34× expansion ratio** (vs. the standard 4×).
Gating buys the most when an FFN is capacity- or selectivity-limited; a
60→2048→60 block is the opposite of that. It may still help via the
multiplicative inductive bias, but the honest prior is "small upside."

## The measurement problem (the hard part)

The delta we report is:

```
observed = arch_effect + GPU_nondeterminism + init_draw_luck + [environment deltas]
```

`GPU_nondeterminism` is not removable: pvcnn's voxelization uses `atomicAdd`
scatters (`pvcnn/modules/functional/src/voxelization/vox.cu`), which are
nondeterministic and compound chaotically over 70 epochs. Its magnitude is
plausibly the same order as the expected gated-FFN effect (sub-mm).

What we **did** remove:

- **Data order.** `DynamicBatchSampler` seeds a *local* `torch.Generator` with
  `seed + epoch` (`datasets/DynamicBatchSampler.py:134`) and takes `seed=0`
  explicitly (`main.py:166`). So data order does **not** depend on the global RNG
  stream, and changing the FFN's parameter shapes does not shift which batches
  the model sees. This is better than a typical seed-confounded comparison — but
  it is **not** the noise floor, and must not be sold as such.
- **TF32.** Ampere enables cuDNN TF32 by default; pinned off in `main.py` so
  precision matches the T4-trained baseline's FP32 semantics.
- **Worker RNG.** Re-added Phase 1's `worker_init_fn` seeding (scene-point
  subsampling was unseeded → ~0.03mm of run-to-run noise).

## Compute plan

Constraints: Friday deadline (~36-48h from Wed 19:40); GCP `GPUS_ALL_REGIONS`
quota = **1** (no parallel arms there); Modal credit ~$30; GCP credit ~3 runs,
expiring in 9 days.

| Arm | Where | GPU | Purpose |
|---|---|---|---|
| **gated** | Modal | A100 | The Friday deliverable. ~8-9h → lands Thu morning with retry slack. |
| **baseline re-run** | GCP | T4 | Pipeline-fidelity check. Same T4 as the original `70.pth`. Uses credits that expire anyway. ~24h, off the critical path. |

`torch 1.13.0+cu117` supports up to sm_86, so **H100 (sm_90) is unusable** without
changing torch (which would change numerics). A100 is the ceiling.

### Why the baseline re-run matters despite reusing 109.71

The original `70.pth` was produced by a machine and codebase that **no longer
exist** (VM deleted 2026-07-15, no snapshot). Comparing the gated arm to 109.71
changes five things at once: FFN, GPU (T4→A100), Python (3.9→3.10), `hik` package
source, and **the filter reconstruction**. The last is decisive: `primary_filterC`
gates the *training* set too (threshold 0.6 on A/B/C), and Phase 0 only gives
verification targets for **D/test** — so the train-side reconstruction is
unverifiable. If it differs at all, `70.pth` trained on a different dataset.

The GCP T4 re-run resolves this: if it lands near 109.71 on the same hardware, it
validates that the rebuilt pipeline (re-reconstructed filters + rebuilt env)
faithfully reproduces the original, which is what licenses the gated-vs-109.71
comparison.

### The cheap partial check: re-evaluating the surviving `70.pth`

The two static verifications (filter counts, param count) prove the *data subset*
and the *architecture* match. Neither proves the rebuilt pipeline produces the
same **number** — that is a different claim, and nothing so far tests it.

`modal_app.py::eval_ckpt` runs the surviving `70.pth` through the rebuilt eval
path (~15 min, ~$0.50). It isolates the **eval-side** half of the question and
answers it *today* rather than in 24h:

- lands near 109.71 → the rebuilt eval path + filters are faithful, and 109.71 is
  still a valid reference to compare the gated arm against;
- lands far off → something in the rebuild drifted, and we learn it **before**
  spending 9 more GPU-hours, not after.

It does **not** subsume the T4 re-run: it reuses the original weights, so it says
nothing about the *training*-side reconstruction (notably `primary_filterC` at
threshold 0.6 on A/B/C, whose counts Phase 0 never recorded). Its A100-vs-T4 delta
also measures the eval-side hardware gap directly, rather than hand-waving it.

## Gate on GPU spend

Before either training run launches, the re-reconstructed filters must reproduce
Phase 0's counts on recording D (`test`): **13,162 admitted of 34,370
fully-present candidates** (21,186 rejected by `primary_filterC`, 22 by
`filterB`/no-others). If they don't match, the comparison to 109.71 is invalid at
the root and no GPU hours should be spent.

## Reporting contract

Report, in this order:

1. gated vs. 109.71 — the number the supervisor asked for, with the five-way
   confound stated plainly.
2. baseline-rerun vs. 109.71 — the pipeline-fidelity check.
3. If `|arch delta|` is smaller than the observed pipeline-fidelity gap, say
   plainly that the result sits inside the noise band. A null result, honestly
   reported, is the correct outcome for a sub-mm-effect lever measured with one
   run per arm — and is itself a finding about what a 24h-per-run budget can and
   cannot resolve.

## Durability requirement

Everything authored here lives in `humof_repro/` in this repo, never only on
compute. The VM deletion destroyed the filters, the preprocess script, and all of
Phase 1's code because they existed nowhere else. The Phase 0/1 *write-ups* are
the only reason recovery was possible — they recorded the filter semantics and
the exact counts needed to verify a reconstruction.
