# Handoff: HUMOF baseline-improvement work

**Date:** 2026-07-15 · **Status:** Phase 0+1 complete and committed; **Phase 2 (gated FFN) training now** — see the Phase 2 section below

## What this project is

This repo (`its-dreOwO/AIL`) vendors the official `hik` data API/eval harness for the NeurIPS 2023
*Humans in Kitchens* (HIK) dataset — 3D multi-person motion forecasting with scene context, four
kitchen recordings (A–D) at 25 Hz. On top of it we build our own forecasting work under
`forecasting/` for an AIL course.

**Timeline of direction changes:**

1. **Own models (siMLPe → HumanMAC), through 2026-06-29.** Built a single-person siMLPe baseline,
   iterated on it three ways (position decoder, horizon-weighted loss, velocity decoder) — all lost
   to the trivial zero-velocity ("freeze last pose") baseline (1.108 MPJPE) due to structural
   mean-collapse at long horizons. Also tried a stochastic HumanMAC (DCT-domain diffusion,
   best-of-K) — trained successfully, found and fixed a DDIM sampler bug (`5e8b644`), but best-of-K
   is an oracle metric and not apples-to-apples with the deterministic bar. This work lives on
   branch `forecasting/simlpe-scene-humanmac` (PR #1, merged). Full state: memory
   `simlpe-baseline-status.md`.
2. **Pivot (2026-06-30): reproduce HUMOF instead of inventing our own model.** HUMOF (Sun et al.,
   ICLR 2026, `github.com/scy639/HUMOF`) is the published SOTA on HIK. Cloned as a sibling repo at
   `/opt/study/ail/HUMOF` (not part of this repo). Paper target (Table 1, "Ours" on HIK, train A–C
   eval D, H=25→F=50, multi-person, 70 epochs): **Path mean 180.7 mm / Pose mean 90.2 mm**.
3. **Repro completed 2026-07-01.** Trained 70 epochs from scratch on the GCP T4 VM (no pretrained
   weights were released). Result on recording D: **path 109.8 mm / pose 66.8 mm** — numerically
   *better* than the paper, but **not comparable**: two functions the eval pipeline depends on,
   `primary_filterC` and `filterB`, were referenced in HUMOF's `dataset_hik.py` but never committed
   to the repo (missing from both HUMOF's git history and the SAST preprocessing repo it depends
   on). We had to reconstruct them from context, which changes which eval windows get admitted.
4. **Current phase (started 2026-07-04): "improve the baseline."** Since our repro's absolute
   numbers can't be honestly compared to Table 1, the course deliverable was reframed as: improve
   *our own* reconstructed-eval-subset number via legitimate, provable techniques, and document the
   reproducibility limitation as a finding in its own right. This is "Approach C" — full design
   rationale in `docs/superpowers/specs/2026-07-04-humof-baseline-improvement-design.md`.

## Why "improve the baseline" is phased the way it is

A single ~24h from-scratch retrain on the T4 VM can't reliably prove a gain smaller than
~1mm (seed/GPU nondeterminism dominates at that scale). So the plan front-loads levers that have
**zero retraining confound** — i.e., wins that are mechanically guaranteed to be real, not noise —
before considering anything that requires a new training run:

- **Phase 0 — subset characterization** (near-free, no training). Quantify how much our
  reconstructed filters shrink/bias the eval set vs. the full dense set. This produces a
  *reproducibility-limitation finding*, independent of whether Phase 1/2 succeed.
- **Phase 1 — inference-time-only wins** (eval-only, deterministic A/B on the *identical* subset,
  reusing the already-trained 70-epoch checkpoint — no retraining). Two independent levers:
  checkpoint weight-averaging, and test-time rotation-averaging (TTA).
- **Phase 2 — one architecture change requiring a ~24h retrain**, upside-only, gated on a
  Phase 0/1-informed decision, and **not started yet** — this is a pending user decision, not a
  blocker for the current deliverable.

## Phase 0 — subset characterization (done, 2026-07-04)

**Finding:** on recording D, of the 34,370 fully-present candidate windows, our reconstructed
`primary_filterC` (motion-magnitude threshold, bbox diagonal ≥ 0.4m) rejects **61.6%**, admitting
only **13,162** into the eval set. `filterB` (social-proximity gate) is nearly a no-op (rejects 22).

**Why it matters:** `primary_filterC`/`filterB` were never committed anywhere in HUMOF's or SAST's
history — we reconstructed the logic and thresholds by inference. Since this single reconstructed
filter decides 61.6% of the eval set's composition, our absolute numbers (109.8/66.8mm) are **not
Table-1-comparable**, and we can't even sign the direction of the gap (the filter keeps
*high-motion*, intuitively harder, windows — yet our numbers are lower than the paper's). This is
reported as the reproducibility-limitation finding for the course.

Full writeup: `docs/results/humof-improve/subset-report.md`.

## Phase 1 — inference-time improvements (done, 2026-07-04)

Two eval-only levers, both run on the *same* 13,162-window subset from the *same* trained
checkpoint (ckpt 70), so any improvement is mechanically real — no seed/GPU confound:

1. **Checkpoint weight-averaging** (`tools/avg_ckpt.py`): elementwise mean of `model_dict` tensors
   across several epoch checkpoints. Best window tried: avg{50,60,70}.
2. **Test-time rotation averaging (TTA4):** run the model 4× per sample, each on a copy rotated
   about the z-axis (0, π/2, π, 3π/2), inverse-rotate each prediction back, and average. The model
   already trains with z-rotation augmentation so this is a legitimate variance-reduction trick
   (4× eval cost, no retraining).

Also fixed an eval-determinism bug along the way: the DataLoader's per-worker numpy RNG (used to
subsample 1000 scene points per window) wasn't seeded, causing ~0.03mm run-to-run noise. Added
`worker_init_fn` seeding so every number below is bit-reproducible.

### Results (recording D, reconstructed-filter subset, deterministic)

| Config | Path mean (mm) | Pose mean (mm) | Δ Path | Δ Pose |
|---|---:|---:|---:|---:|
| Baseline (ckpt 70) | 109.712 | 66.806 | — | — |
| Checkpoint avg{50,60,70} | 109.359 | 66.558 | −0.35 | −0.25 |
| **TTA4 (ckpt 70)** | **107.146** | 66.267 | **−2.57** | −0.54 |
| TTA4 + avg{50,60,70} | 107.393 | **66.228** | −2.32 | **−0.58** |

**Headline: TTA4 is the clean win — path −2.57mm (−2.3%), pose −0.54mm — retraining-free and
exactly reproducible.** Its benefit grows with horizon (0.5s: −0.94mm → 2s: −5.34mm path), which
tracks with prediction variance being largest at long horizons — exactly where averaging helps
most. TTA and checkpoint-averaging don't stack cleanly on path (combining them is *worse* than TTA
alone, 107.39 vs 107.15) — TTA already captures most of the available gain there.

Full writeup + per-horizon breakdown: `docs/results/humof-improve/phase1-results.md`. Raw per-run
CSVs also in this directory (`err_det_base70.csv`, `err_det_tta4_70.csv`, etc).

## Phase 2 — IN FLIGHT as of 2026-07-15 (gated FFN)

**The lever changed.** Not iterative refinement (the old candidate below) — the supervisor directed
a **gated FFN** in the transformer layer, and that is what is running. Design spec:
`docs/superpowers/specs/2026-07-15-humof-gated-ffn-design.md`. Deadline **Fri 2026-07-17**.

**The old GCP VM `hik-simlpe-train` was deleted** (no snapshot), destroying the filters, the
preprocess script, and all of Phase 1's code — they existed nowhere else. Everything was rebuilt
from the Phase 0/1 write-ups, which is the only reason recovery was possible. Everything authored
since lives in `humof_repro/` **in this repo**.

### Two verifications that underpin the deliverable

1. **Filter reconstruction is behaviorally exact.** Rebuilt `primary_filterC`/`filterB` reproduce
   all three Phase 0 counters on D: 69,290 absent / 284 partial / **13,162 admitted**.
2. **Param-matching is exact.** Original run logged `10.01099M` params; gated logs `10.01377M` =
   **+2,780** exactly as the 2/3-hidden math predicts (+0.028%). Confirms SwiGLU is live and
   nothing else in the rebuild drifted.

### Arms

| Arm | Where | Status |
|---|---|---|
| **gated** (deliverable) | Modal A100, `--detach` | Training. ~9.3 min/epoch → ~11h, ETA ~08:00 Thu. |
| **baseline re-run** | GCP T4 `hik-humof-base`, `us-west4-b` | Provisioning. ~24h, off critical path. |

**Modal preempts containers.** Observed once ~8 min in (KeyboardInterrupt at the dataloader poll +
`Runner terminated`), and Modal re-runs the function from the top — which restarted training at
epoch 0. Fixed: the Modal wrapper autodetects the newest checkpoint and sets `HUMOF_CP_ITER`,
feeding `main.py`'s existing integer resume path (restores model+optimizer+scheduler, runs
`range(cp_iter, num_epoch)`). Do **not** use `main.py`'s own `'auto'` branch — it `int()`s the env
var before it can match `'auto'`, then blocks on `input()`. Save interval cut 10→2 epochs
(`HUMOF_SAVE_INTERVAL`) so a preemption costs ~24 min instead of ~2h; multiples of 10 are still
saved, so Phase 1's checkpoint-averaging set stays a subset.

**Known confound, to report honestly:** resume restores model/opt/scheduler but *not* the global
torch RNG, so a restarted run's dropout/aug draws diverge from an uninterrupted one. This joins the
already-documented confound list and sits inside the noise band — not worth chasing byte-identity.

## What's left / open decision (superseded lever, kept for reference)

The pre-supervisor Phase 2 candidate was iterative multi-person refinement (the paper's own
future-work #3): the HHI module only sees other people's *history*, never their *predicted future*.
Alternative: velocity/acceleration input features. Neither is being pursued right now.

Phase 1 alone (−2.57mm path, exactly provable) remains a solid, defensible deliverable on its own.

## Where things live

- **Design spec:** `docs/superpowers/specs/2026-07-04-humof-baseline-improvement-design.md`
- **Results (this dir):** `subset-report.md` (Phase 0), `phase1-results.md` (Phase 1), raw CSVs
- **HUMOF repro repo:** `/opt/study/ail/HUMOF` (sibling repo, not part of `AIL`) — has the trained
  checkpoints (`checkpoints/hik-releaseV0.1/`), eval code (`main.py`, `datasets/dataset_hik.py`),
  reconstructed filters (`utils.py`, thresholds in `confs.py`), TTA hook (`main.py`, backups
  `.bak_pretta`/`.bak_improve`), and `tools/avg_ckpt.py`
- **Phase 2 repro kit:** `humof_repro/` in this repo — `modal_app.py` (the Modal app: image,
  volumes, `train`/`eval_ckpt`/`verify_filters`), `humof_addons.py` (reconstructed filters),
  `preprocess_hik_direct.py`, `patches/phase2-gated-ffn.patch` (every HUMOF-side edit, since HUMOF
  is a separate checkout), `gcp_grab_t4.sh`, `gcp_setup_baseline.sh`, `README.md`.
- **GCP VM:** `hik-simlpe-train` is **DELETED** — do not look for it. The current one is
  `hik-humof-base` (T4, zone `us-west4-b`), reachable as `ssh hikvm`. Keep it **stopped** between
  uses to avoid billing; the external IP changes on every start, so refresh `~/.ssh/config`
  HostName from `gcloud compute instances describe hik-humof-base --zone=us-west4-b
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`. T4 capacity is scarce on a credit
  project — every zone can be simultaneously out of stock; `gcp_grab_t4.sh` cycles all 51 T4 zones
  until one has capacity (it won in `us-west4-b` on round 1).
- **Earlier (superseded) own-model work:** branch `forecasting/simlpe-scene-humanmac` / PR #1 in
  this repo, under `forecasting/`. Kept for reference; not part of the current HUMOF-repro direction.
