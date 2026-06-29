# Scene-Conditioned Forecasting Design

**Date:** 2026-06-28
**Status:** Draft for review
**Scope:** deterministic Tier 3 scene-feature conditioning for the existing single-person forecasting pipeline

## Motivation

The single-person siMLPe Tier 1 line has now been exhausted:

| horizon | zero-velocity | siMLPe position | weighted position | velocity output |
|---------|--------------:|----------------:|------------------:|----------------:|
| overall | **1.108**     | 1.188           | 1.184             | 1.197           |
| @1s     | **0.520**     | 0.607           | 0.572             | 0.537           |
| @5s     | **1.254**     | 1.351           | 1.348             | 1.318           |
| @10s    | **1.422**     | 1.472           | 1.462             | 1.707           |

The failure mode is structural mean-collapse: deterministic single-person motion
over 10 seconds is too ambiguous under position-space MPJPE, and zero-velocity is
hard to beat without extra information. The next iteration must add information
zero-velocity does not use. Scene context is the lowest-risk first Tier 3 step:
people in kitchens often move toward objects such as sinks, cupboards, whiteboards,
coffee machines, and sittable objects.

## Goals

- Add a compact scene-conditioning path that works in both training and
  `Evaluator.execute3d` callbacks.
- Preserve the existing `SiMLPe` baseline, checkpoint compatibility, and comparison
  commands.
- Reuse the vendored `Kitchen`/`KitchenObject` APIs without editing `hik/`.
- Keep local work limited to code edits and synthetic/unit tests. Full data
  training/evaluation stays on the GCP T4 VM.
- Produce a clear VM experiment: train scene-conditioned siMLPe on A+B+C+D,
  evaluate on dataset A, and compare directly to zero-velocity overall MPJPE 1.108.

## Non-Goals

- No HumanMAC/diffusion implementation in this iteration.
- No social attention over other people yet.
- No hand-authored action labels or goal-object labels.
- No changes to vendored `hik/`, `testdata/`, `documentation/`, or notebooks.
- No local real-data training or evaluation.

## Existing Interfaces

Training currently materializes normalized single-person windows:

- `build_windows_for_dataset(...)` loads a `Scene`, slices raw `scene.poses3d`, and
  stores normalized `[WINDOW, 29, 3]` windows.
- `WindowDataset.__getitem__` returns `(x, y)` where `x` is `[250, 87]` and `y` is
  `[250, 87]`.
- `train_one_epoch` and `eval_loss` assume batches are exactly `(x, y)`.

Evaluation currently normalizes one person at a time:

- `make_simlpe_callback(model, device)` receives `inp["Poses3d_in"]`,
  `inp["Masks_in"]`, `inp["kitchen"]`, `inp["frames_in"]`, and `inp["pids"]`.
- It backfills masks, normalizes the selected person at frame `N_IN - 1`, calls
  `model(x)`, and denormalizes the prediction.

Scene geometry is available through vendored APIs:

- Training: `Scene.kitchen`.
- Eval: `inp["kitchen"]`.
- Object query: `kitchen.get_environment(frame, ignore_oob=True, use_pointcloud=False)`.
- Each returned `EnvironmentObject` has `name`, `label` as a 13D one-hot type vector,
  `isbox`, and `location`.

## Scene Feature Contract

Create `forecasting/scene_features.py` with a pure extractor:

```python
SCENE_FEATURE_DIM = 13 + 4

def extract_scene_features(kitchen, frame: int, pose3d: np.ndarray) -> np.ndarray:
    ...
```

The first implementation uses the nearest non-out-of-bound object at the last
observed frame:

- `pose3d`: one person's last observed world pose, shape `[29, 3]`.
- Person anchor: pelvis/root proxy, initially `pose3d.mean(axis=0)` to avoid relying
  on undocumented joint ids. This can later be replaced with a known pelvis joint
  once verified.
- For each object from `kitchen.get_environment(frame, ignore_oob=True, use_pointcloud=False)`:
  - box location `[8, 3]`: object center is mean of corners.
  - cylinder location `[4]`: object center is `location[:3]`.
  - direction is object center minus person anchor in XY.
  - distance is `sqrt(dx^2 + dy^2)`.
- Select the closest object by XY distance.
- Output vector:
  - 13D object type one-hot.
  - normalized `dx`, `dy`, and `distance` divided by a fixed 10-meter scale.
  - `present` flag, 1.0 when at least one object exists, 0.0 otherwise.

If there is no kitchen or no valid object, return zeros. This keeps fallback behavior
defined for synthetic tests and future ablations.

Rationale: nearest-object features are cheap, deterministic, and available in both
training and eval. They are weaker than an explicit goal label, but they establish
the full scene-conditioned data/model/callback path with minimal moving parts.

## Data Pipeline

Add scene-conditioned builders without replacing the current window cache:

```python
def build_scene_windows_for_dataset(dataset, stepsize, test_json_path, max_windows=None):
    ...

def build_or_load_scene_windows(datasets, stepsize, test_json_path):
    ...

class SceneWindowDataset(torch.utils.data.Dataset):
    def __getitem__(self, i):
        return x, scene_features, y
```

The cache will be a compressed `.npz` with:

- `windows`: `[N, WINDOW, 29, 3]`, same normalized pose windows as today.
- `scene`: `[N, SCENE_FEATURE_DIM]`, scene features computed from the raw world pose
  at `start + N_IN - 1` before normalization.

Use a distinct cache name, for example:

```text
scene_windows_ABCD_s50_v1.npz
```

This avoids corrupting or reinterpreting the existing `windows_ABCD_s50.npy` cache.

## Model

Add a conditioned model that preserves the temporal backbone:

```python
class SceneConditionedSiMLPe(nn.Module):
    def __init__(..., scene_dim=SCENE_FEATURE_DIM, scene_hidden=128, output_mode="position"):
        self.backbone = SiMLPe-compatible temporal stack
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, scene_hidden),
            nn.GELU(),
            nn.Linear(scene_hidden, pose_dim),
        )
```

The simplest injection is to add the encoded scene vector to every input frame before
the existing DCT/temporal MLP path:

```python
h = motion_fc_in(x) + scene_encoder(scene_features).unsqueeze(1)
```

Then decode with the same `position` or `velocity` output modes already supported by
`SiMLPe`. The recommended first run should use `position`, because the 2026-06-28
velocity run improved short/mid horizon but introduced severe long-horizon drift.

Alternative injection points considered:

- Concatenate scene features to every pose frame: simple but changes the temporal
  input dimension and complicates DCT assumptions.
- Add scene embedding after DCT: viable, but less direct and harder to reason about
  in tests.
- Predict goal object as an auxiliary head: likely useful later, but requires label
  design and a multi-loss training objective.

## Training CLI

Keep `forecasting.train` backward-compatible and add a model selector:

```bash
python -m forecasting.train \
  --model scene-simlpe \
  --datasets A B C D --stepsize 50 --epochs 80 \
  --lr 5e-4 --vel-weight 0.2 --horizon-floor 0.2 \
  --output-mode position
```

Implementation detail:

- `--model simlpe` uses the current `build_or_load_windows` and `WindowDataset`.
- `--model scene-simlpe` uses `build_or_load_scene_windows` and `SceneWindowDataset`.
- `train_one_epoch`/`eval_loss` will accept either `(x, y)` or `(x, scene, y)` batches
  through a small helper that calls `model(x)` or `model(x, scene)` based on batch
  shape.
- Save to a distinct default checkpoint path for scene-conditioned runs, for example
  `forecasting/cache/scene_simlpe.pt`.

## Evaluation CLI and Callback

Add a scene-conditioned callback:

```python
def make_scene_simlpe_callback(model, device="cpu"):
    ...
```

For each person:

1. Backfill and normalize the person's input pose exactly like `make_simlpe_callback`.
2. Compute scene features from the original world-space last observed pose:
   - `kitchen = inp["kitchen"]`
   - `frame = inp["frames_in"][-1]`
   - `pose3d = poses_in[-1, pid]`
3. Call `model(x, scene_features)`.
4. Denormalize and write the prediction.
5. If normalization or scene extraction fails, fall back to zero-velocity for that
   person rather than failing the whole callback.

Extend `forecasting.evaluate`:

```bash
python -m forecasting.evaluate \
  --dataset A --model scene-simlpe \
  --ckpt forecasting/cache/scene_simlpe.pt \
  --output-mode position
```

## Testing Plan

Local tests are synthetic-only.

Scene feature tests:

- Returns zeros when `kitchen is None`.
- Selects nearest object by XY distance.
- Handles box and cylinder object locations.
- Returns fixed shape `SCENE_FEATURE_DIM` and finite `float32` values.

Data tests:

- `SceneWindowDataset` splits `(windows, scene)` into `(x, scene, y)`.
- Cache/build helper naming is distinct from existing `windows_*.npy`.
- Existing `WindowDataset` behavior remains unchanged.

Model tests:

- `SceneConditionedSiMLPe(x, scene)` returns `[B, 250, 87]`.
- Changing scene features changes output when scene encoder weights are nonzero.
- `output_mode="position"` still anchors on the last frame when the output head is zero.

Callback tests:

- Fake kitchen + fake scene model produces finite `[250, P, 29, 3]`.
- The scene-conditioned callback passes a `[1, SCENE_FEATURE_DIM]` tensor to the model.
- Existing `make_simlpe_callback` tests still pass.

CLI/load tests:

- `load_model(..., model_name="scene-simlpe")` constructs the conditioned model.
- `train(..., model_name="scene-simlpe")` instantiates the conditioned dataset/model
  path when given synthetic arrays.

Verification commands:

```bash
/opt/study/.venv/bin/python -m pytest tests/ -q -m "not slow"
/opt/study/.venv/bin/python -m unittest forecasting.test_losses -v
```

Remote VM commands after local tests pass:

```bash
gcloud compute instances start hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b

gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && python3 -m forecasting.train --model scene-simlpe --datasets A B C D --stepsize 50 --epochs 80 --lr 5e-4 --vel-weight 0.2 --horizon-floor 0.2 --output-mode position'

gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && python3 -m forecasting.evaluate --dataset A --model scene-simlpe --ckpt forecasting/cache/scene_simlpe.pt --output-mode position'

gcloud compute instances stop hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b
```

## Success Criteria

- Local synthetic/unit tests pass.
- Remote scene-conditioned training completes on the T4 VM.
- Remote evaluation reports overall and horizon MPJPE.
- Hard bar for an acceptable result remains overall MPJPE `< 1.108`.
- If the scene-conditioned run still loses overall, record it as diagnostic and move
  to the next Tier 3/Tier 4 method rather than tuning the same architecture blindly.

## Risks and Mitigations

- **Nearest object is not always the goal.** This is expected. The first objective is
  to establish a correct scene-conditioned path. A later version can add goal-object
  supervision or top-K objects.
- **Feature scale mismatch.** Normalize distances/directions by a fixed meter scale
  and keep a `present` flag.
- **Cache memory pressure.** Scene features add only `[N, 17]`; the expensive part
  remains the existing pose windows.
- **Long-horizon drift.** Use `output_mode position` for the first scene run and keep
  velocity mode as an ablation, not the default.
- **VM dataset path drift.** The current VM uses `$HOME/Humans_in_Kitchen`; verify
  before training and stop the VM after every run.
