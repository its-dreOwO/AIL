# HUMOF baseline improvement — design (Approach C)

**Date:** 2026-07-04
**Context:** AIL course. HUMOF (ICLR 2026) reproduced on HIK; 70-epoch train gives path 109.8 / pose 66.8 mm on recording D. This spec plans *improving our own subset number* + documenting a genuine limitation.

## Goal & non-goals

- **Goal:** lower our HIK-D MPJPE (path/pose) on our exact eval subset, with a defensible A/B, and produce one solid limitation finding — deliverable for the course.
- **Non-goal:** beat / match paper Table 1 (180.7 / 90.2). Our reconstructed `primary_filterC`/`filterB` gate a non-standard subset, so absolute numbers are not comparable. Only *relative* gains on the *same* subset are claimed.

## Key constraint (drives the whole plan)

A single from-scratch retrain is ~24h on the T4, and we cannot afford multiple seeds. So a <~1mm delta from one retrain is **indistinguishable from seed/GPU nondeterminism**. Therefore we front-load levers that have **zero retraining confound** (inference-time, checkpoint-reuse) where a small gain is provably real, and treat any architecture retrain as *upside*, never as the thing the deliverable depends on.

## Phases

### Phase 0 — subset characterization (near-free, finding)
Count how many test cases our `primary_filterC`/`filterB` admit on recording D, broken down per action + total, vs. the raw dense count (all valid person-windows). If the admitted subset is materially smaller/easier, that is a concrete, honest reproducibility limitation — likely sufficient as a standalone finding.
- **Where:** VM (per no-local-heavy-runs), or a cheap standalone script over `Scene` if it fits in local RAM (verify first; default to VM).
- **Output:** a small table (action → admitted / total, mean displacement of admitted vs. rejected).

### Phase 1 — inference-time wins (eval-only, provable)
No retraining, so every number is a clean A/B on the identical subset.
- **(a) Checkpoint averaging / ensembling.** Ckpts 10–70 are on the VM. Try (i) weight-averaging the last N checkpoints into one model, and (ii) prediction-ensembling (mean of per-checkpoint outputs). Report both vs. the single-70 baseline.
- **(b) Test-time rotation averaging.** The model trains with rotation augmentation; average predictions over K rotated copies of the input (rotate in, inverse-rotate out, mean). Eval-only cost (~K× one eval run).
- **Output:** baseline vs. each lever vs. combined, on path & pose, per horizon (0.5/1/1.5/2s) + mean.

### Phase 2 — one architecture change (~24h retrain, upside only)
Lever selected *after* Phase 0/1, informed by where the error concentrates.
- **Leading candidate:** iterative multi-person refinement (paper future-work #3). HHI currently ingests only other people's *history* `Y(k)`; add a second pass where HHI also sees the first-pass *predicted future* of others. One retrain.
- **Alternative:** if abrupt-motion failure cases dominate the residual error, add velocity/acceleration channels to the motion encoder input.
- **Honesty guardrail:** report the delta with an explicit note that a single-run gain within ~1–2mm is not distinguishable from training variance; only claim a clear win if the margin exceeds that.

## Deliverable shape
A short report/section: (1) repro state + why numbers aren't paper-comparable, (2) Phase 0 subset finding, (3) Phase 1 provable inference-time improvement table, (4) Phase 2 architectural attempt with an honest variance caveat.

## Risks
- Phase 1 gains could be ~0 (checkpoint 70 may already be near-optimal). Mitigation: Phase 0 finding + honest reporting still constitute a valid deliverable.
- VM IP changes on every stop/start — refresh `~/.ssh/config` HostName before each session.
