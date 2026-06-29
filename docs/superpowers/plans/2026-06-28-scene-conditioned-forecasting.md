# Scene-Conditioned Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic scene-conditioned siMLPe path that uses compact kitchen-object features during training and evaluation, then run the VM experiment against the zero-velocity MPJPE bar.

**Architecture:** Add a pure `forecasting.scene_features` extractor, a scene-window dataset/cache path, and `SceneConditionedSiMLPe`, while preserving the existing unconditioned `SiMLPe` path. Training and evaluation gain a `scene-simlpe` model selector; local verification remains synthetic/unit-only, and real training/eval runs on the GCP T4 VM.

**Tech Stack:** Python, NumPy, PyTorch, pytest/unittest, vendored HIK `Kitchen`/`Evaluator`, GCP VM `hik-simlpe-train`.

---

## File Structure

- Create `forecasting/scene_features.py`: pure feature extraction from `Kitchen`/fake kitchens and world-space last pose.
- Modify `forecasting/data.py`: add `SceneWindowDataset`, `build_scene_windows_for_dataset`, and `build_or_load_scene_windows`.
- Modify `forecasting/model.py`: add `SceneConditionedSiMLPe` while keeping `SiMLPe` behavior.
- Modify `forecasting/train.py`: add model selector, scene-aware batch dispatch, and scene checkpoint default.
- Modify `forecasting/callbacks.py`: add `make_scene_simlpe_callback`.
- Modify `forecasting/evaluate.py`: add `scene-simlpe` model option and load/callback dispatch.
- Modify `forecasting/README.md` and `forecasting/GOAL.md`: document commands and result slot.
- Create/modify tests in `tests/test_scene_features.py`, `tests/test_data.py`, `tests/test_model.py`, `tests/test_train.py`, `tests/test_callbacks.py`, and `tests/test_evaluate.py`.

Do not modify vendored `hik/`, `testdata/`, `documentation/`, or notebooks.

---

### Task 1: Scene Feature Extractor

**Files:**
- Create: `tests/test_scene_features.py`
- Create: `forecasting/scene_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_features.py`:

```python
import numpy as np

from forecasting.scene_features import SCENE_FEATURE_DIM, extract_scene_features


class FakeObject:
    def __init__(self, location, label, isbox=True):
        self.location = np.asarray(location, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.float32)
        self.isbox = isbox


class FakeKitchen:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def get_environment(self, frame, ignore_oob=True, use_pointcloud=False):
        self.calls.append((frame, ignore_oob, use_pointcloud))
        return self.objects


def _label(index):
    out = np.zeros(13, dtype=np.float32)
    out[index] = 1.0
    return out


def test_scene_features_zero_without_kitchen():
    pose = np.ones((29, 3), dtype=np.float32)
    features = extract_scene_features(None, frame=10, pose3d=pose)
    assert features.shape == (SCENE_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.allclose(features, 0.0)


def test_scene_features_selects_nearest_box_by_xy_distance():
    pose = np.zeros((29, 3), dtype=np.float32)
    far_box = np.array(
        [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [5.0, 1.0, 0.0],
         [5.0, 0.0, 1.0], [6.0, 0.0, 1.0], [6.0, 1.0, 1.0], [5.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    near_box = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    kitchen = FakeKitchen(
        [
            FakeObject(far_box, _label(3), isbox=True),
            FakeObject(near_box, _label(10), isbox=True),
        ]
    )

    features = extract_scene_features(kitchen, frame=42, pose3d=pose)

    assert kitchen.calls == [(42, True, False)]
    assert np.allclose(features[:13], _label(10))
    assert np.allclose(features[13:17], [0.15, 0.05, np.sqrt(2.5) / 10.0, 1.0])


def test_scene_features_handles_cylinder_location():
    pose = np.zeros((29, 3), dtype=np.float32)
    kitchen = FakeKitchen([FakeObject([0.0, 3.0, 0.0, 0.4], _label(4), isbox=False)])

    features = extract_scene_features(kitchen, frame=7, pose3d=pose)

    assert np.allclose(features[:13], _label(4))
    assert np.allclose(features[13:17], [0.0, 0.3, 0.3, 1.0])
    assert np.isfinite(features).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_scene_features.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'forecasting.scene_features'`.

- [ ] **Step 3: Implement the extractor**

Create `forecasting/scene_features.py`:

```python
import numpy as np


OBJECT_TYPE_DIM = 13
SCENE_GEOM_DIM = 4
SCENE_FEATURE_DIM = OBJECT_TYPE_DIM + SCENE_GEOM_DIM
SCENE_DISTANCE_SCALE = 10.0


def _object_center(obj) -> np.ndarray:
    loc = np.asarray(obj.location, dtype=np.float32)
    if getattr(obj, "isbox", False):
        return loc.reshape(-1, 3).mean(axis=0)
    return loc[:3]


def extract_scene_features(kitchen, frame: int, pose3d: np.ndarray) -> np.ndarray:
    features = np.zeros(SCENE_FEATURE_DIM, dtype=np.float32)
    if kitchen is None:
        return features

    pose = np.asarray(pose3d, dtype=np.float32)
    if pose.shape != (29, 3):
        return features
    anchor = pose.mean(axis=0)

    try:
        objects = kitchen.get_environment(
            frame, ignore_oob=True, use_pointcloud=False
        )
    except Exception:
        return features

    best = None
    best_dist = None
    for obj in objects:
        center = _object_center(obj)
        delta_xy = center[:2] - anchor[:2]
        dist = float(np.linalg.norm(delta_xy))
        if best_dist is None or dist < best_dist:
            best = (obj, delta_xy, dist)
            best_dist = dist

    if best is None:
        return features

    obj, delta_xy, dist = best
    label = np.asarray(obj.label, dtype=np.float32).reshape(-1)
    n = min(OBJECT_TYPE_DIM, label.shape[0])
    features[:n] = label[:n]
    features[13] = delta_xy[0] / SCENE_DISTANCE_SCALE
    features[14] = delta_xy[1] / SCENE_DISTANCE_SCALE
    features[15] = dist / SCENE_DISTANCE_SCALE
    features[16] = 1.0
    return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_scene_features.py -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add forecasting/scene_features.py tests/test_scene_features.py
git commit -m "feat(forecasting): add scene feature extractor"
```

---

### Task 2: Scene Window Dataset

**Files:**
- Modify: `tests/test_data.py`
- Modify: `forecasting/data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
from forecasting.scene_features import SCENE_FEATURE_DIM
from forecasting.data import SceneWindowDataset, scene_windows_cache_path


def test_scene_window_dataset_splits_pose_and_scene_features():
    windows = np.random.default_rng(1).standard_normal(
        (3, config.WINDOW, 29, 3)
    ).astype(np.float32)
    scene = np.random.default_rng(2).standard_normal(
        (3, SCENE_FEATURE_DIM)
    ).astype(np.float32)

    ds = SceneWindowDataset(windows, scene)
    x, s, y = ds[1]

    assert x.shape == (config.N_IN, config.POSE_DIM)
    assert s.shape == (SCENE_FEATURE_DIM,)
    assert y.shape == (config.N_OUT, config.POSE_DIM)
    assert torch.is_tensor(x) and x.dtype == torch.float32
    assert torch.is_tensor(s) and s.dtype == torch.float32
    assert torch.is_tensor(y) and y.dtype == torch.float32


def test_scene_window_dataset_rejects_mismatched_lengths():
    windows = np.zeros((2, config.WINDOW, 29, 3), dtype=np.float32)
    scene = np.zeros((3, SCENE_FEATURE_DIM), dtype=np.float32)

    try:
        SceneWindowDataset(windows, scene)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scene_windows_cache_path_is_distinct_from_pose_window_cache():
    path = scene_windows_cache_path(["A", "B"], stepsize=50)

    assert path.endswith("scene_windows_AB_s50_v1.npz")
    assert "windows_AB_s50.npy" not in path
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_data.py::test_scene_window_dataset_splits_pose_and_scene_features tests/test_data.py::test_scene_window_dataset_rejects_mismatched_lengths tests/test_data.py::test_scene_windows_cache_path_is_distinct_from_pose_window_cache -q
```

Expected: FAIL with import errors for `SceneWindowDataset` and `scene_windows_cache_path`.

- [ ] **Step 3: Implement dataset and cache path**

Modify imports in `forecasting/data.py`:

```python
from forecasting.scene_features import SCENE_FEATURE_DIM, extract_scene_features
```

Add below `WindowDataset`:

```python
class SceneWindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray, scene_features: np.ndarray):
        if len(windows) != len(scene_features):
            raise ValueError("windows and scene_features must have the same length")
        if scene_features.shape[1] != SCENE_FEATURE_DIM:
            raise ValueError(f"bad scene feature shape {scene_features.shape}")
        self.windows = windows
        self.scene_features = scene_features

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        window = self.windows[i].reshape(WINDOW, POSE_DIM)
        x = torch.tensor(window[:N_IN], dtype=torch.float32)
        scene = torch.tensor(self.scene_features[i], dtype=torch.float32)
        y = torch.tensor(window[N_IN:], dtype=torch.float32)
        return x, scene, y


def scene_windows_cache_path(datasets, stepsize) -> str:
    tag = "".join(datasets) + f"_s{stepsize}"
    return os.path.join(config.cache_dir(), f"scene_windows_{tag}_v1.npz")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_data.py -q -m "not slow"
```

Expected: PASS for non-slow data tests.

- [ ] **Step 5: Commit**

```bash
git add forecasting/data.py tests/test_data.py
git commit -m "feat(forecasting): add scene window dataset"
```

---

### Task 3: Scene Window Builders

**Files:**
- Modify: `tests/test_data.py`
- Modify: `forecasting/data.py`

- [ ] **Step 1: Write the failing builder tests**

Append to `tests/test_data.py`:

```python
def test_build_or_load_scene_windows_loads_npz_cache(monkeypatch, tmp_path):
    cached_windows = np.zeros((2, config.WINDOW, 29, 3), dtype=np.float32)
    cached_scene = np.ones((2, SCENE_FEATURE_DIM), dtype=np.float32)
    cache_path = tmp_path / "scene_windows_AB_s50_v1.npz"
    np.savez_compressed(cache_path, windows=cached_windows, scene=cached_scene)

    monkeypatch.setattr(
        "forecasting.data.scene_windows_cache_path",
        lambda datasets, stepsize: str(cache_path),
    )

    windows, scene = build_or_load_scene_windows(["A", "B"], 50, TEST_JSON)

    assert np.array_equal(windows, cached_windows)
    assert np.array_equal(scene, cached_scene)


def test_build_or_load_scene_windows_saves_npz_cache(monkeypatch, tmp_path):
    built_windows = np.zeros((1, config.WINDOW, 29, 3), dtype=np.float32)
    built_scene = np.ones((1, SCENE_FEATURE_DIM), dtype=np.float32)
    cache_path = tmp_path / "scene_windows_A_s50_v1.npz"

    monkeypatch.setattr(
        "forecasting.data.scene_windows_cache_path",
        lambda datasets, stepsize: str(cache_path),
    )
    monkeypatch.setattr(
        "forecasting.data.build_scene_windows_for_dataset",
        lambda dataset, stepsize, test_json_path: (built_windows, built_scene),
    )

    windows, scene = build_or_load_scene_windows(["A"], 50, TEST_JSON)

    assert np.array_equal(windows, built_windows)
    assert np.array_equal(scene, built_scene)
    loaded = np.load(cache_path)
    assert np.array_equal(loaded["windows"], built_windows)
    assert np.array_equal(loaded["scene"], built_scene)
```

Also update the import block in `tests/test_data.py`:

```python
from forecasting.data import (
    SceneWindowDataset,
    WindowDataset,
    build_or_load_scene_windows,
    build_windows_for_dataset,
    forbidden_frames,
    scene_windows_cache_path,
    window_is_clean,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_data.py::test_build_or_load_scene_windows_loads_npz_cache tests/test_data.py::test_build_or_load_scene_windows_saves_npz_cache -q
```

Expected: FAIL with import error for `build_or_load_scene_windows`.

- [ ] **Step 3: Implement builders**

Add to `forecasting/data.py`:

```python
def build_scene_windows_for_dataset(dataset, stepsize, test_json_path, max_windows=None):
    forbidden = forbidden_frames(dataset, test_json_path)
    scene = Scene.load_from_paths(
        dataset=dataset,
        person_path=poses_path(),
        scene_path=scenes_path(),
        smplx_path=dirname(smplx_path()),
    )
    poses3d = scene.poses3d
    masks = scene.masks
    starts = frame_splits(scene.frames, length=WINDOW, stepsize=stepsize)
    n_person = masks.shape[1]

    windows = []
    scene_rows = []
    for start in starts:
        sl = slice(start, start + WINDOW)
        if masks[sl].shape[0] != WINDOW:
            continue
        for person_idx in range(n_person):
            if not window_is_clean(start, masks[sl, person_idx], forbidden):
                continue
            block = poses3d[sl, person_idx][:, None]
            try:
                normed, _ = normalize3d(block, frame=N_IN - 1)
            except ValueError:
                continue
            last_frame = start + N_IN - 1
            features = extract_scene_features(
                scene.kitchen,
                frame=last_frame,
                pose3d=poses3d[last_frame, person_idx],
            )
            windows.append(normed[:, 0])
            scene_rows.append(features)
            if max_windows is not None and len(windows) >= max_windows:
                return (
                    np.asarray(windows, dtype=np.float32),
                    np.asarray(scene_rows, dtype=np.float32),
                )
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(scene_rows, dtype=np.float32),
    )


def build_or_load_scene_windows(datasets, stepsize, test_json_path):
    path = scene_windows_cache_path(datasets, stepsize)
    if os.path.exists(path):
        cached = np.load(path)
        return cached["windows"], cached["scene"]
    parts = [
        build_scene_windows_for_dataset(dataset, stepsize, test_json_path)
        for dataset in datasets
    ]
    windows = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
    scene = np.concatenate([p[1] for p in parts], axis=0).astype(np.float32)
    np.savez_compressed(path, windows=windows, scene=scene)
    return windows, scene
```

- [ ] **Step 4: Run data tests**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_data.py -q -m "not slow"
```

Expected: PASS for non-slow data tests.

- [ ] **Step 5: Commit**

```bash
git add forecasting/data.py tests/test_data.py
git commit -m "feat(forecasting): build scene-conditioned window cache"
```

---

### Task 4: Scene-Conditioned Model

**Files:**
- Modify: `tests/test_model.py`
- Modify: `forecasting/model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
from forecasting.scene_features import SCENE_FEATURE_DIM
from forecasting.model import SceneConditionedSiMLPe


def test_scene_conditioned_forward_shape():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=2, scene_dim=SCENE_FEATURE_DIM
    )
    x = torch.randn(4, 250, 87)
    scene = torch.randn(4, SCENE_FEATURE_DIM)

    y = model(x, scene)

    assert y.shape == (4, 250, 87)
    assert torch.isfinite(y).all()


def test_scene_conditioned_output_changes_with_scene_features():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=1, scene_dim=SCENE_FEATURE_DIM
    )
    x = torch.randn(1, 250, 87)
    scene_a = torch.zeros(1, SCENE_FEATURE_DIM)
    scene_b = torch.ones(1, SCENE_FEATURE_DIM)

    y_a = model(x, scene_a)
    y_b = model(x, scene_b)

    assert not torch.allclose(y_a, y_b)


def test_scene_conditioned_position_mode_anchors_on_last_frame_when_head_zero():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=1, scene_dim=SCENE_FEATURE_DIM
    )
    torch.nn.init.zeros_(model.motion_fc_out.weight)
    torch.nn.init.zeros_(model.motion_fc_out.bias)
    x = torch.randn(2, 250, 87)
    scene = torch.randn(2, SCENE_FEATURE_DIM)

    y = model(x, scene)

    assert torch.allclose(y, x[:, -1:, :].expand(-1, 250, -1), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_model.py::test_scene_conditioned_forward_shape tests/test_model.py::test_scene_conditioned_output_changes_with_scene_features tests/test_model.py::test_scene_conditioned_position_mode_anchors_on_last_frame_when_head_zero -q
```

Expected: FAIL with import error for `SceneConditionedSiMLPe`.

- [ ] **Step 3: Implement the model**

Modify `forecasting/model.py`:

```python
class SceneConditionedSiMLPe(nn.Module):
    """siMLPe temporal backbone with a per-window scene embedding."""

    def __init__(
        self,
        t_in=250,
        t_out=250,
        pose_dim=87,
        n_blocks=4,
        scene_dim=17,
        scene_hidden=128,
        output_mode="position",
    ):
        super().__init__()
        if t_in != t_out:
            raise AssertionError("this baseline uses square DCT over equal lengths")
        if output_mode not in {"position", "velocity"}:
            raise ValueError("output_mode must be 'position' or 'velocity'")
        self.t_in = t_in
        self.t_out = t_out
        self.pose_dim = pose_dim
        self.output_mode = output_mode

        dct_m, idct_m = get_dct_matrix(t_in)
        self.register_buffer("dct", torch.tensor(dct_m, dtype=torch.float32))
        self.register_buffer("idct", torch.tensor(idct_m, dtype=torch.float32))

        self.motion_fc_in = nn.Linear(pose_dim, pose_dim)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, scene_hidden),
            nn.GELU(),
            nn.Linear(scene_hidden, pose_dim),
        )
        self.blocks = nn.ModuleList([TemporalMLPBlock(t_in) for _ in range(n_blocks)])
        self.motion_fc_out = nn.Linear(pose_dim, pose_dim)

    def forward(self, x, scene_features):
        last = x[:, -1:, :]
        scene_h = self.scene_encoder(scene_features).unsqueeze(1)
        h = self.motion_fc_in(x) + scene_h
        h = torch.einsum("tk,bkc->btc", self.dct, h)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        h = torch.einsum("tk,bkc->btc", self.idct, h)
        h = self.motion_fc_out(h)
        if self.output_mode == "velocity":
            return last + torch.cumsum(h, dim=1)
        return h + last
```

- [ ] **Step 4: Run model tests**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add forecasting/model.py tests/test_model.py
git commit -m "feat(forecasting): add scene-conditioned simlpe model"
```

---

### Task 5: Scene-Aware Training Path

**Files:**
- Modify: `tests/test_train.py`
- Modify: `forecasting/train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
from forecasting.scene_features import SCENE_FEATURE_DIM


def test_train_one_epoch_passes_scene_batch_to_model():
    class SceneModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.seen_scene_shape = None

        def forward(self, x, scene):
            self.seen_scene_shape = tuple(scene.shape)
            return x * self.scale

    x = torch.randn(2, config.N_IN, config.POSE_DIM)
    scene = torch.randn(2, SCENE_FEATURE_DIM)
    y = torch.randn(2, config.N_OUT, config.POSE_DIM)
    loader = [(x, scene, y)]
    model = SceneModel()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_one_epoch(model, loader, optim, "cpu", vel_weight=0.0)

    assert model.seen_scene_shape == (2, SCENE_FEATURE_DIM)


def test_train_uses_scene_model_and_dataset(monkeypatch):
    seen = {}

    class FakeSceneModel:
        def __init__(self, **kwargs):
            seen["model"] = kwargs["output_mode"]

        def to(self, device):
            return self

        def parameters(self):
            return []

    class FakeDataset:
        def __init__(self, windows, scene):
            seen["dataset_shapes"] = (windows.shape, scene.shape)

        def __len__(self):
            return 4

    monkeypatch.setattr("forecasting.train.SceneConditionedSiMLPe", FakeSceneModel)
    monkeypatch.setattr("forecasting.train.SceneWindowDataset", FakeDataset)
    monkeypatch.setattr("forecasting.train.train_one_epoch", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("forecasting.train.eval_loss", lambda *args, **kwargs: 1.0)

    class FakeOptim:
        def __init__(self, params, lr):
            pass

    class FakeSched:
        def __init__(self, optim, T_max):
            pass

        def step(self):
            pass

    monkeypatch.setattr("torch.optim.Adam", FakeOptim)
    monkeypatch.setattr("torch.optim.lr_scheduler.CosineAnnealingLR", FakeSched)

    windows = np.zeros((4, config.WINDOW, config.N_JOINTS, 3), dtype=np.float32)
    scene = np.zeros((4, SCENE_FEATURE_DIM), dtype=np.float32)
    train(
        (windows, scene),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_name="scene-simlpe",
        output_mode="position",
        out_path=None,
    )

    assert seen["model"] == "position"
    assert seen["dataset_shapes"] == (windows.shape, scene.shape)
```

Update `tests/test_train.py` imports:

```python
from forecasting.train import train, train_one_epoch
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_train.py::test_train_one_epoch_passes_scene_batch_to_model tests/test_train.py::test_train_uses_scene_model_and_dataset -q
```

Expected: FAIL because `train_one_epoch` unpacks only `(x, y)` and `train()` lacks `model_name`.

- [ ] **Step 3: Implement scene-aware training dispatch**

Modify imports in `forecasting/train.py`:

```python
from forecasting.data import (
    SceneWindowDataset,
    WindowDataset,
    build_or_load_scene_windows,
    build_or_load_windows,
)
from forecasting.model import SceneConditionedSiMLPe, SiMLPe
```

Add helper functions near the top:

```python
def _split_batch(batch, device):
    if len(batch) == 2:
        x, y = batch
        return x.to(device), None, y.to(device)
    x, scene, y = batch
    return x.to(device), scene.to(device), y.to(device)


def _predict(model, x, scene):
    if scene is None:
        return model(x)
    return model(x, scene)
```

Update both loops:

```python
for batch in loader:
    x, scene, y = _split_batch(batch, device)
    optim.zero_grad()
    pred = _predict(model, x, scene)
```

and in `eval_loss`:

```python
for batch in loader:
    x, scene, y = _split_batch(batch, device)
    pred = _predict(model, x, scene)
```

Update `train()` signature:

```python
def train(
    windows,
    *,
    epochs,
    batch_size=256,
    lr=5e-4,
    val_frac=0.05,
    vel_weight=0.2,
    horizon_floor=0.2,
    device=None,
    n_blocks=4,
    output_mode="velocity",
    model_name="simlpe",
    seed=0,
    out_path="__default__",
):
```

Inside `train()` choose dataset/model/checkpoint:

```python
if out_path == "__default__":
    ckpt_name = "scene_simlpe.pt" if model_name == "scene-simlpe" else "simlpe.pt"
    out_path = os.path.join(config.cache_dir(), ckpt_name)

if model_name == "scene-simlpe":
    pose_windows, scene_features = windows
    dataset = SceneWindowDataset(pose_windows, scene_features)
    model = SceneConditionedSiMLPe(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
        output_mode=output_mode,
    ).to(device)
elif model_name == "simlpe":
    dataset = WindowDataset(windows)
    model = SiMLPe(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
        output_mode=output_mode,
    ).to(device)
else:
    raise ValueError("model_name must be 'simlpe' or 'scene-simlpe'")
```

Remove the old unconditional `dataset = WindowDataset(windows)` and unconditional `model = SiMLPe(...)` blocks.

Update CLI:

```python
parser.add_argument("--model", choices=["simlpe", "scene-simlpe"], default="simlpe")
```

and in `main()`:

```python
if args.model == "scene-simlpe":
    windows = build_or_load_scene_windows(args.datasets, args.stepsize, test_json)
else:
    windows = build_or_load_windows(args.datasets, args.stepsize, test_json)
...
model_name=args.model,
```

- [ ] **Step 4: Run train tests**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_train.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add forecasting/train.py tests/test_train.py
git commit -m "feat(forecasting): train scene-conditioned simlpe"
```

---

### Task 6: Scene-Conditioned Callback

**Files:**
- Modify: `tests/test_callbacks.py`
- Modify: `forecasting/callbacks.py`

- [ ] **Step 1: Write failing callback test**

Append to `tests/test_callbacks.py`:

```python
from forecasting.scene_features import SCENE_FEATURE_DIM
from forecasting.callbacks import make_scene_simlpe_callback


class RecordingSceneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_shape = None

    def forward(self, x, scene):
        self.scene_shape = tuple(scene.shape)
        return x


def test_scene_simlpe_callback_shape_and_scene_tensor():
    class FakeObject:
        isbox = False
        location = np.array([2.0, 0.0, 0.0, 0.5], dtype=np.float32)
        label = np.eye(13, dtype=np.float32)[10]

    class FakeKitchen:
        def get_environment(self, frame, ignore_oob=True, use_pointcloud=False):
            return [FakeObject()]

    model = RecordingSceneModel()
    cb = make_scene_simlpe_callback(model, device="cpu")
    inp = _make_input(P=1)
    inp["kitchen"] = FakeKitchen()

    out = cb(inp)

    assert out.shape == (250, 1, 29, 3)
    assert np.isfinite(out).all()
    assert model.scene_shape == (1, SCENE_FEATURE_DIM)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_callbacks.py::test_scene_simlpe_callback_shape_and_scene_tensor -q
```

Expected: FAIL with import error for `make_scene_simlpe_callback`.

- [ ] **Step 3: Implement callback**

Modify imports in `forecasting/callbacks.py`:

```python
from forecasting.scene_features import extract_scene_features
```

Add below `make_simlpe_callback`:

```python
def make_scene_simlpe_callback(model, device="cpu"):
    model = model.to(device)
    model.eval()

    def callback(inp: dict) -> np.ndarray:
        poses_in = np.copy(inp["Poses3d_in"])
        masks_in = np.copy(inp["Masks_in"])
        n_out = inp["n_out"]
        n_person = poses_in.shape[1]
        fallback = zero_velocity_callback(inp)
        pred_world = np.empty((n_out, n_person, 29, 3), dtype=np.float32)
        kitchen = inp.get("kitchen")
        frames_in = inp.get("frames_in", [])
        frame = frames_in[-1] if len(frames_in) else N_IN - 1

        filled, _ = backfill_masked(poses_in, masks_in)
        for pid in range(n_person):
            block = filled[:, pid : pid + 1]
            try:
                normed, norm_params = normalize3d(block, frame=N_IN - 1)
                scene = extract_scene_features(kitchen, frame=frame, pose3d=poses_in[-1, pid])
            except Exception:
                pred_world[:, pid] = fallback[:, pid]
                continue

            x = normed[:, 0].reshape(1, N_IN, POSE_DIM)
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32, device=device)
                st = torch.tensor(scene[None], dtype=torch.float32, device=device)
                yt = model(xt, st)
            pred_normed = yt.cpu().numpy().reshape(n_out, 1, 29, 3)
            pred_world[:, pid : pid + 1] = denormalize3d(pred_normed, norm_params)
        return pred_world.astype(np.float32)

    return callback
```

- [ ] **Step 4: Run callback tests**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_callbacks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add forecasting/callbacks.py tests/test_callbacks.py
git commit -m "feat(forecasting): add scene-conditioned callback"
```

---

### Task 7: Evaluation Model Selector

**Files:**
- Modify: `tests/test_evaluate.py`
- Modify: `forecasting/evaluate.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_evaluate.py`:

```python
def test_load_model_uses_scene_model(monkeypatch):
    seen = {}

    class FakeSceneModel:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def load_state_dict(self, state):
            seen["state"] = state

    monkeypatch.setattr(evaluate, "SceneConditionedSiMLPe", FakeSceneModel)
    monkeypatch.setattr(evaluate.torch, "load", lambda *args, **kwargs: {"ok": True})

    model = evaluate.load_model(
        "checkpoint.pt", model_name="scene-simlpe", output_mode="position"
    )

    assert model is not None
    assert seen["kwargs"]["output_mode"] == "position"
    assert seen["state"] == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_evaluate.py::test_load_model_uses_scene_model -q
```

Expected: FAIL because `load_model()` lacks `model_name` or `SceneConditionedSiMLPe`.

- [ ] **Step 3: Implement evaluate dispatch**

Modify imports in `forecasting/evaluate.py`:

```python
from forecasting.callbacks import (
    make_scene_simlpe_callback,
    make_simlpe_callback,
    zero_velocity_callback,
)
from forecasting.model import SceneConditionedSiMLPe, SiMLPe
```

Replace `load_model` with:

```python
def load_model(ckpt, n_blocks=4, output_mode="velocity", device="cpu", model_name="simlpe"):
    model_kwargs = dict(
        t_in=config.N_IN,
        t_out=config.N_OUT,
        pose_dim=config.POSE_DIM,
        n_blocks=n_blocks,
        output_mode=output_mode,
    )
    if model_name == "scene-simlpe":
        model = SceneConditionedSiMLPe(**model_kwargs)
    elif model_name == "simlpe":
        model = SiMLPe(**model_kwargs)
    else:
        raise ValueError("model_name must be 'simlpe' or 'scene-simlpe'")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model
```

Update CLI model choices:

```python
parser.add_argument("--model", choices=["zerovel", "simlpe", "scene-simlpe"], default="zerovel")
```

Update dispatch:

```python
if args.model == "zerovel":
    callback = zero_velocity_callback
else:
    model = load_model(
        args.ckpt,
        n_blocks=args.n_blocks,
        output_mode=args.output_mode,
        device=device,
        model_name=args.model,
    )
    if args.model == "scene-simlpe":
        callback = make_scene_simlpe_callback(model, device=device)
    else:
        callback = make_simlpe_callback(model, device=device)
```

- [ ] **Step 4: Run evaluate tests**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/test_evaluate.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add forecasting/evaluate.py tests/test_evaluate.py
git commit -m "feat(forecasting): evaluate scene-conditioned simlpe"
```

---

### Task 8: Docs, CLI Verification, and Full Local Tests

**Files:**
- Modify: `forecasting/README.md`
- Modify: `forecasting/GOAL.md`

- [ ] **Step 1: Update runbook commands**

Add a "Train scene-conditioned siMLPe" section to `forecasting/README.md`:

```markdown
## Train scene-conditioned siMLPe

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen"
python -m forecasting.train \
  --model scene-simlpe \
  --datasets A B C D --stepsize 50 --epochs 80 \
  --lr 5e-4 --vel-weight 0.2 --horizon-floor 0.2 \
  --output-mode position
```

This writes `forecasting/cache/scene_simlpe.pt` and uses the separate
`scene_windows_ABCD_s50_v1.npz` cache.

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen"
python -m forecasting.evaluate \
  --dataset A --model scene-simlpe \
  --ckpt forecasting/cache/scene_simlpe.pt \
  --output-mode position
```
```

In `forecasting/GOAL.md`, update the current plan to say the active next experiment
is scene-feature conditioned siMLPe with `--model scene-simlpe`.

- [ ] **Step 2: Run CLI help checks**

Run:

```bash
/opt/study/.venv/bin/python -m forecasting.train --help
/opt/study/.venv/bin/python -m forecasting.evaluate --help
```

Expected:

- train help includes `--model {simlpe,scene-simlpe}`.
- evaluate help includes `{zerovel,simlpe,scene-simlpe}`.

- [ ] **Step 3: Run full local synthetic/unit verification**

Run:

```bash
/opt/study/.venv/bin/python -m pytest tests/ -q -m "not slow"
/opt/study/.venv/bin/python -m unittest forecasting.test_losses -v
```

Expected:

- pytest exits 0 with all non-slow tests passing.
- unittest exits 0 with all loss tests passing.

- [ ] **Step 4: Commit**

```bash
git add forecasting/README.md forecasting/GOAL.md
git commit -m "docs(forecasting): document scene-conditioned experiment"
```

---

### Task 9: Remote VM Training and Evaluation

**Files:**
- Modify after run: `forecasting/README.md`
- Modify after run: `forecasting/GOAL.md`

- [ ] **Step 1: Verify VM status**

Run:

```bash
gcloud compute instances describe hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --format='value(status)'
```

Expected: `TERMINATED` or `RUNNING`. If `TERMINATED`, start it in Step 2.

- [ ] **Step 2: Start VM if needed**

Run only if Step 1 returned `TERMINATED`:

```bash
gcloud compute instances start hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b
```

Expected: command exits 0.

- [ ] **Step 3: Copy local branch files to VM or push/pull**

Use the repo's preferred delivery path. If copying directly, copy only changed
`forecasting/`, `tests/`, and `docs/superpowers/` files into `~/hik`; do not copy
unrelated dirty files.

Example direct copy:

```bash
gcloud compute scp \
  forecasting/scene_features.py forecasting/data.py forecasting/model.py \
  forecasting/train.py forecasting/callbacks.py forecasting/evaluate.py \
  forecasting/README.md forecasting/GOAL.md \
  hik-simlpe-train:~/hik/forecasting/ \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b

gcloud compute scp \
  tests/test_scene_features.py tests/test_data.py tests/test_model.py \
  tests/test_train.py tests/test_callbacks.py tests/test_evaluate.py \
  hik-simlpe-train:~/hik/tests/ \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b
```

Expected: commands exit 0.

- [ ] **Step 4: Run remote unit tests**

Run:

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --command='cd ~/hik && python3 -m pytest tests/ -q -m "not slow" && python3 -m unittest forecasting.test_losses -v'
```

Expected: tests pass remotely before training starts.

- [ ] **Step 5: Train scene-conditioned model**

Run:

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && python3 -u -m forecasting.train --model scene-simlpe --datasets A B C D --stepsize 50 --epochs 80 --lr 5e-4 --vel-weight 0.2 --horizon-floor 0.2 --output-mode position'
```

Expected: command exits 0 and writes `forecasting/cache/scene_simlpe.pt`.

- [ ] **Step 6: Evaluate scene-conditioned model**

Run:

```bash
gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --command='cd ~/hik && export HIK_DATA="$HOME/Humans_in_Kitchen" && python3 -m forecasting.evaluate --dataset A --model scene-simlpe --ckpt forecasting/cache/scene_simlpe.pt --output-mode position'
```

Expected: output includes overall mean MPJPE and @1s/@5s/@10s values.

- [ ] **Step 7: Stop VM**

Run:

```bash
gcloud compute instances stop hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --quiet
```

Then verify:

```bash
gcloud compute instances describe hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --format='value(status)'
```

Expected: `TERMINATED`.

- [ ] **Step 8: Record result**

Update `forecasting/README.md` and `forecasting/GOAL.md` with a result section
that includes the exact values printed by Step 6. The section must include:

- heading: `Result (2026-06-28 scene-conditioned siMLPe)`.
- config: `scene-simlpe`, `output_mode position`, `lr 5e-4`, `vel_weight 0.2`,
  `horizon_floor 0.2`, A+B+C+D, stepsize 50, 80 epochs, Tesla T4.
- table columns: `horizon`, `zero-velocity`, `scene-conditioned siMLPe`.
- rows: `overall`, `@1s`, `@5s`, and `@10s`.
- conclusion: if overall is below `1.108`, mark it as meeting the hard bar; if not,
  mark it diagnostic and move to the next Tier 3/Tier 4 approach.

- [ ] **Step 9: Commit result docs**

```bash
git add forecasting/README.md forecasting/GOAL.md
git commit -m "docs(forecasting): record scene-conditioned result"
```

---

## Final Verification Checklist

- [ ] Local non-slow pytest passes.
- [ ] Local `forecasting.test_losses` unittest passes.
- [ ] Remote non-slow pytest/unittest passes before training.
- [ ] Remote training writes `forecasting/cache/scene_simlpe.pt`.
- [ ] Remote evaluation reports overall and horizon MPJPE.
- [ ] VM is verified `TERMINATED` after the run.
- [ ] `forecasting/README.md` and `forecasting/GOAL.md` record the result.
- [ ] If overall MPJPE is `< 1.108`, the forecasting goal can be considered achieved for the deterministic track. Otherwise keep the goal active and proceed to the next method.
