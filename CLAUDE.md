# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hik` is the official **data API + evaluation harness** for the NeurIPS 2023 dataset *"Humans in Kitchens"* (multi-person 3D human-motion forecasting with scene context). It is **not a model** — it ships no neural network. Researchers bring their own forecasting model and plug it into the evaluation protocol. The codebase only loads the dataset, visualizes it, and scores predictions.

The dataset is four kitchen recordings labelled **A, B, C, D**, captured at **25 Hz**. Almost every API takes a `dataset` ("A".."D") and a `frame` (int).

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

`examples.py` and `animate.py` default the data dir to `/home/dre/Downloads/hik_dataset/data`, overridable with the `HIK_DATA` env var. The dataset layout (all paths derive from `data/`):

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
