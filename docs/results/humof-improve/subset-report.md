# Phase 0 — HUMOF HIK eval-subset characterization

**Date:** 2026-07-04 · recording **D** (the eval split), measured on the VM's runnable HUMOF copy.

## What gates the eval subset

`DatasetHik('test')` slides a window (`t_total = H25+F50 = 75` frames) every `STEP_test = 41`
frames over recording D, then applies filters. Counts from an instrumented run:

| Stage | Count | Note |
|---|---:|---|
| Absent in window (`present_agg` false) | 69,290 | person not in frame at all — not real candidates |
| Partial presence (`present_sum < t_total`) | 284 | present but not for the full 75 frames |
| **Fully-present target windows (candidates)** | **34,370** | the meaningful denominator |
| — rejected by `primary_filterC` (low motion) | 21,186 | **61.6% of candidates** |
| — rejected by `filterB`/no-other (`len_others==0`) | 22 | negligible |
| **Admitted (final eval set, `len(dataset)`)** | **13,162** | **38.3% of candidates** |

**The eval subset is gated almost entirely by `primary_filterC`** — a motion-magnitude
threshold that discards ~62% of fully-present windows. `filterB` (social proximity) barely
touches the *target* set here (only 22 windows end up with zero qualifying others); it mainly
shapes which *other* people are fed to the HHI module.

## The exact reconstructed filter parameters (recording D / `test`)

From `datasets/dataset_preprocess/hik/confs.py` (values we reconstructed — see below):

- `STEP_test = 41`, `STEP_train = 17`
- `primary_filterC`: keep if most-moving-joint axis-aligned **bbox diagonal ≥ 0.4 m** over the
  window (`THRES__primary_filterC_test = 0.4`; train uses 0.6). Skips near-static targets.
- `filterB`: an "other" counts only if its root comes within **8.0 m** of the target root at some
  frame (`THRES_filterB = 8.0`; train 9.9).

## Why this is a reproducibility limitation (the finding)

`primary_filterC` / `filterB` were **never committed to the HUMOF repo** (absent from its history
and from SAST). We **reconstructed** them (bodies in `utils.py`, thresholds in `confs.py`), so:

1. **The dominant gate is our reconstruction.** `primary_filterC` alone decides 61.6% of the
   candidate set. Its threshold (0.4 m) is our guess, not the authors'. A different threshold →
   a materially different subset **size and composition**. There is also an unused
   `PRIMARY_FILTER_C2` sub-flag whose original logic we do not know.
2. **Absolute numbers are therefore not comparable to Table 1** (paper: path 180.7 / pose 90.2;
   ours: 109.76 / 66.79). The subset that produced each is different.
3. **We cannot even sign the gap.** `primary_filterC` retains *high-motion* windows (larger
   displacement, intuitively *harder* in absolute mm), yet our path error is *lower* than the
   paper's. So the subset difference does not straightforwardly explain the direction of the gap —
   meaning our lower number must not be read as "beating" HUMOF.

**Conclusion:** treat the repro as *pipeline-validated, absolute numbers not comparable*. All
improvement claims in Phase 1 are made as **relative A/Bs on this exact 13,162-window subset**,
where the identical filter applies to baseline and variant alike — so the reconstruction cancels
out and the deltas are meaningful even though the absolute level is not paper-comparable.

## Reproduce

Instrumented counters added to `datasets/dataset_hik.py` (VM; backup `.bak_improve`). Rerun:
```bash
python3 -c "import globals_; globals_.TRAIN=False
from conf import *; from datasets.dataset_hik import DatasetHik
ds=DatasetHik('test'); print('LEN', len(ds))"   # prints __ct_* + SUBSET_ADMIT_RATE
```
