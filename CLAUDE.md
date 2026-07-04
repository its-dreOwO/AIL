# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hik` is the official **data API + evaluation harness** for the NeurIPS 2023 dataset *"Humans in Kitchens"* (multi-person 3D human-motion forecasting with scene context). It is **not a model** — it ships no neural network. Researchers bring their own forecasting model and plug it into the evaluation protocol. The codebase only loads the dataset, visualizes it, and scores predictions.

**This repo is `github.com/its-dreOwO/AIL`** (origin points there, fresh history). It **vendors** the upstream `hik` harness (by Tanke et al., NeurIPS 2023) as a **read-only dependency** and adds our own forecasting models under `forecasting/`. Treat `hik/`, `testdata/`, `documentation/`, `notebooks/` as vendored — **do not edit them** (incl. the `eval/mpjpe.py::calc_mpjpe` stub; our scoring lives in `forecasting/metrics.py`).

The dataset is four kitchen recordings labelled **A, B, C, D**, captured at **25 Hz**. Almost every API takes a `dataset` ("A".."D") and a `frame` (int).

## Our forecasting work (the `forecasting/` package)

We build forecasting models on top of the vendored harness. Roadmap: **(1) single-person siMLPe baseline** → (2) multi-person social attention → (3) scene-aware → (4) generative. **The target to beat is the zero-velocity ("freeze last pose") baseline: overall MPJPE 1.108** (@1s 0.520, @5s 1.254, @10s 1.422).

- Pose rep: canonical `hik.transforms.utils.normalize3d` at frame 249 (last observed), `denormalize3d` for scoring. Eval callback contract: in `[250,P,29,3]` → out `[250,P,29,3]`. Metric: `forecasting/metrics.py` (`calc_mpjpe`, plus `calc_best_of_k_mpjpe` for the stochastic track).
- **Compute: never run dataset/training/eval locally** (14 GB RAM OOMs). All heavy runs go to the GCP T4 VM `hik-simlpe-train` (project `project-b9a4f950-85a4-48f0-9ee`, zone `asia-southeast1-b`); start it to train, **stop it after**. Reuse the `forecasting/cache/windows_ABCD_s50.npy` window cache (no rebuild).

### Progress (updated 2026-06-29)

- **Tier 1 — siMLPe: implemented, runs, but loses to zero-velocity.** Three iterations (position decoder, horizon-weighted loss, velocity decoder) all land overall ≥ 1.18. Root cause is structural mean-collapse at the 10 s horizon, not a bug — the pipeline is verified correct.
- **Tier 3 (partial) — nearest-object scene-conditioned siMLPe: implemented, also loses** (overall 1.1845). Nearest-object features alone aren't enough.
- **Phase B — HumanMAC stochastic best-of-K: trained (converged), sampler bug found + fixed, RE-EVALUATING on the T4 (2026-06-29).** Whole-sequence DCT-domain diffusion (transformer denoiser, masked-completion conditioning), evaluated via best-of-K(50). Modules: `forecasting/diffusion.py`, `forecasting/humanmac.py`, `forecasting/eval_bok.py`; `--model humanmac` in `train.py`/`evaluate.py`. Training converged ~epoch 450 (val ~0.015 denoising MSE), stopped at epoch 480/500; checkpoint `forecasting/cache/humanmac.pt`. ⚠️ **best-of-K is an ORACLE metric** (uses GT to pick the best of K samples) — always reported alongside the single-sample number; a best-of-K win is not apples-to-apples with the 1.108 deterministic bar.
  - **DDIM sampler bug (fixed, commit `5e8b644`):** cosine schedule gives `alphas_cumprod[999]≈2.4e-9`, so the DDIM `x0=(X-√(1-abar)·eps)/√abar` divided by ~5e-5 and **exploded x0 ~20000×** on the first step, knocking the trajectory off-manifold → samples came out ~100× the data scale (pre-fix eval was garbage: best-of-50 ~39 "m", horizon error inverted). Fix = static thresholding (clip predicted x0 to ±50; real DCT coeffs max ~39) in `ddim_sample`/`HumanMAC.x0_clip`. **No retraining needed** — training was sound, only sampling was broken. Verified: `|fut|` absmax 195→3.8.
  - **Branch/PR:** all forecasting work (siMLPe + scene + HumanMAC + this fix) is on branch `forecasting/simlpe-scene-humanmac` → **PR #1** (`github.com/its-dreOwO/AIL/pull/1`). The old `single-person-simlpe-baseline` branch was renamed/superseded and its stale remote copy deleted.
- Specs/plans: `docs/superpowers/specs/2026-06-28-humanmac-stochastic-forecasting-design.md`, `docs/superpowers/plans/2026-06-28-humanmac-stochastic-forecasting.md`. Full status: `forecasting/GOAL.md`.

### Current direction (updated 2026-06-30): reproduce HUMOF (ICLR 2026)

Pivoted from our own models to **reproducing HUMOF** (`github.com/scy639/HUMOF`, the SOTA on HIK), cloned as a sibling repo at `/opt/study/ail/HUMOF` (not under this repo). Paper: `/home/dre/Downloads/2506.03753v3.pdf`. **Target = paper Table 1 "Ours" on HIK: Path mean 180.7 mm / Pose mean 90.2 mm** (train A–C, eval D, H=25→F=50, multi-person, 70 epochs). Full state: memory `humof-repro-plan.md`.

**Repro done (2026-07-01):** 70-epoch train on HIK → **path 109.8 / pose 66.8 mm on recording D** (see `results/hik-releaseV0.1/err.csv`). NOT Table-1-comparable — our reconstructed `primary_filterC`/`filterB` gate a different (easier/smaller) eval subset.

### Improve-the-baseline plan (updated 2026-07-04, Approach "C"): for the AIL course

Goal: **improve our own HIK-D subset number** (not beat the paper — absolute numbers aren't comparable) + document a real limitation. Key constraint: a single ~24h from-scratch retrain **can't prove a <~1mm gain** (seed/GPU variance), so front-load levers with **zero retraining confound**. Phased so a bad training run can't sink the deliverable. Spec: `docs/superpowers/specs/2026-07-04-humof-baseline-improvement-design.md`.

- **Phase 0 — subset characterization (near-free finding).** Count test cases our `primary_filterC`/`filterB` admit on D (per action + total) vs. raw dense count. If materially smaller/easier → concrete reproducibility limitation. Run on VM (no local heavy runs).
- **Phase 1 — inference-time wins (eval-only, provable A/B on identical subset).** (a) **Checkpoint averaging/ensembling** over ckpts 10–70 (still on VM); (b) **test-time rotation averaging** (model already trains with rot-aug). No retraining → any gain is real.
- **Phase 2 — one architecture change (~24h retrain, upside only).** Lever chosen AFTER Phase 0/1 findings. Leading candidate = paper's own future-work #3: **iterative multi-person refinement** (HHI only sees others' *history*, never their *predicted future* — feed a first-pass prediction back into HHI). Alt if abrupt-motion cases dominate error: velocity/acceleration input features.

**Phase 0+1 DONE (2026-07-04), VM stopped. Results in `docs/results/humof-improve/`.** Phase 0: reconstructed `primary_filterC` gates 61.6% of target windows on D (13,162/34,370) → absolute numbers not Table-1-comparable. Phase 1 (deterministic A/B, same subset): **TTA4 test-time rotation-averaging wins path 107.15 (−2.57mm) / pose 66.27 (−0.54mm)** vs det baseline 109.71/66.81, retraining-free + exactly reproducible; checkpoint avg{50,60,70} smaller (−0.35/−0.25). Fixed eval nondeterminism (unseeded DataLoader-worker numpy, ~0.03mm scene-resampling noise) via `worker_init_fn`. **Phase 2 (retrain) is a pending user decision** — Phase 1 alone is a solid deliverable.

- **VM access:** `ssh hikvm` (direct alias, key `~/.ssh/google_compute_engine`). ⚠️ External IP changes on every VM stop/start — refresh `HostName` in `~/.ssh/config` from `gcloud compute instances describe hik-simlpe-train --zone=asia-southeast1-b --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`.
- **Dataset already on VM** at `~/Humans_in_Kitchen/{poses,scenes,body_models}` (no upload needed).
- **Env (`humof` conda, py3.9.19):** torch **1.13.0+cu117** (the repo's `torch==1.12.0+cu117` pin is impossible — that wheel never existed; its tv0.14.0/ta0.13.0 pins actually require 1.13.0), numpy 1.24.3, nvcc 11.7 via conda cuda-toolkit, `pip install -e ~/hik`. pvcnn JIT-compiles only after `ln -sfn $CONDA_PREFIX/lib $CONDA_PREFIX/lib64` (fixes `-lcudart`) + `pip install ninja`. Run with `TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=$CONDA_PREFIX`.
- **Preprocessing: SKIP SAST.** Repo's `preprocess.py` is broken (windows then `assert len(tus)==1`). `DatasetHik` only needs `hik_preprocessed/H25F50/hik_{A..D}/tus.pkl` = `(subseq[P,n_frames,29,3], kid, present[P,n_frames])` = raw dense `Scene.poses3d`/`masks` rearranged. Generate via `~/preprocess_hik_direct.py` (uses only `hik.data.scene.Scene`).
- **BUG — missing functions:** `primary_filterC`/`filterB` are referenced by `dataset_hik.py`/`dataset_hoim3.py` but were never committed (not in HUMOF history nor SAST). Reconstructed in `utils.py` (motion-displacement / social-proximity filters). These gate the eval subset → absolute numbers are NOT like-for-like with Table 1 (our subset evals ~easier).
- **Config edits in `~/HUMOF`:** `conf0.py` `DATASET_name='hik'`; `conf2.py` `CUDA_VISIBLE_DEVICES='0'` (ships as `'1'` — no GPU on a 1-GPU VM!), `num_workers=8`.
- **Run:** `setsid bash ~/run_train.sh > ~/train_hik.log`. ~20 min/epoch, GPU-bound, ~24 h for 70 ep. Ckpts every 10 ep → `checkpoints/hik-releaseV0.1/`; eval → `results/hik-releaseV0.1/err.csv`.

## Environment & commands

A pre-made venv lives at `/opt/study/.venv` (symlinked as `./.venv`); it has all deps installed including `torch`.

```bash
source .venv/bin/activate

# import-only check (no dataset needed)
python smoke_test.py
# full check: loads data, renders one frame to smoke_render.png
python smoke_test.py /home/dre/Downloads/hik_dataset/data

# explore the dataset (stats, activities, find, scene, pose, render)
python examples.py stats
python examples.py find A drink          # frames where someone does an activity
python examples.py render A              # auto-picks a busy frame -> render_A_<frame>.png

# animate a time window with per-person movement trails -> anim_<kitchen>_<start>.mp4
python animate.py A                      # auto-picks busiest ~1min window
python animate.py A 104750 1500 5        # kitchen start n_frames step

# unit tests (only data/utils.py has them)
python -m unittest hik.data.utils
python -m unittest hik.data.utils.TestDataUtils.test_splits   # single test
```

Install from scratch (other machines): `pip install -e .` then `pip install torch`. For long-term evaluation only, also `pip install git+https://github.com/jutanke/ndms.git`.

## Dataset path convention

The **full dataset** lives on a mounted drive at `/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen` (note: path has spaces — pass it via the `HIK_DATA` env var, never hardcode). `forecasting/config.py` defaults `HIK_DATA` to this path. The three subdirs sit directly under it (no `data/` wrapper):

- `poses/{dataset}_{pid}_{seqid}.npz` — one file per person-sequence
- `scenes/{dataset}_scene/{object}.npy` + `.json` — per-frame scene geometry
- `body_models/SMPLX_NEUTRAL.npz` — SMPL-X parametric body model

## Architecture

Four sub-packages under `hik/`:

**`hik.data`** — loading and indexing, the core of the API.
- `PersonSequence` (`person_sequence.py`) wraps one `.npz`. Per-frame arrays: `poses3d` (n×29×3), `smpl` (n×21×3 axis-angle), `transforms` (n×6 = 3 translation + 3 axis-angle rotation), `act` (n×82 multi-hot activities), `betas` (10 shape params), `frames` (contiguous int indices, +1 each step). Quaternion conversions (`get_smpl_as_quaterions`, `get_transforms_as_quaternion`) are **cached to tempfiles in `gettempdir()`** keyed by a content hash — stale caches can silently persist across data changes.
- `PersonSequences` loads *all* `.npz` and builds three lookup dicts; the `(dataset, frame)` index is what makes `get_frame(dataset, frame)` return everyone visible at a moment. `get_block3d` / `Scene.get_splits` densify into `n_frames × n_person × 29 × 3` tensors **plus a validity mask** (people come and go, so masks matter everywhere).
- `Kitchen` / `KitchenObject` (`kitchen.py`) load scene geometry. **Geometry is dynamic and frame-indexed** via `frame_multiplier`: `0` = static object (reuse index 0), `2` = annotated at 12.5 Hz so frames are halved to map onto 25 Hz pose frames. `EnvironmentObject.is_inside(pts3d)` does box/cylinder collision (used to tell which object a hand touches). Objects are boxes (8×3 corners) or cylinders (3D center + radius, always on z=0 floor).
- `Scene` (`scene.py`) materializes a whole recording into dense `poses3d/masks/activities/transforms/smpls/betas` arrays and slices fixed-length training windows.
- `utils.py` — `frames2segments` (list → contiguous `(start,end)` runs) and `get_splits` (sliding windows that never cross gaps). **Only file with unit tests.**

**`hik.transforms`** — axis-angle ↔ quaternion (`rotation.py`) and `normalize()` (re-center/re-orient a clip to canonical pose via left/right hip joints). Standard preprocessing.

**`hik.data.smpl.Body`** — wraps `smplx.SMPLX`; turns `(betas, pose, translation, rotation)` into a posed mesh/joints. `find_transformation` is Kabsch/SVD rigid alignment into SMPL canonical space.

**`hik.eval`** — the benchmark. `Evaluator.execute3d(callback_fn, n_in=250, n_out=250)` is the **standard forecasting protocol**: for each test case in `testdata/test.json` (`{A..D} → {action → [{pid, frame}]}`, 6 actions: walking, sitting_down, whiteboard, sink, cupboard, coffee), it gathers **10 s of past motion for all people present**, calls `callback_fn` (which receives past poses, masks, kitchen, pids), and compares the predicted 10 s future to ground truth. Metric primitive is `mpjpe.mean_per_joint_l2_distance`. Results are pickled via `eval.save_results`.

## Sharp edges

- `eval/mpjpe.py::calc_mpjpe` is a **stub** (`pass` + commented body) in this checkout; the working primitive is `mean_per_joint_l2_distance`.
- `torch` is required (imported in `smoke_test.py` and `smpl.py`) but is **not** named in `setup.py` `install_requires` — it arrives transitively via `smplx`. Don't assume it's pinned.
- Long-term eval (`eval/longterm.py`) imports the external `ndms` package, which is not a `setup.py` dependency.
- The 29-joint skeleton topology and left/right joint coloring live in `hik/vis/pose.py::get_meta`.
