import json
import os
from os.path import dirname

import numpy as np
import torch

from hik.data import Scene, get_splits as frame_splits
from hik.transforms.utils import normalize3d
from forecasting import config
from forecasting.config import (
    N_IN,
    N_OUT,
    POSE_DIM,
    WINDOW,
    poses_path,
    scenes_path,
    smplx_path,
)
from forecasting.scene_features import SCENE_FEATURE_DIM, extract_scene_features


def forbidden_frames(dataset: str, test_json_path: str) -> set:
    with open(test_json_path, "r") as f:
        data = json.load(f)
    frames = set()
    for entries in data.get(dataset, {}).values():
        for entry in entries:
            frame = entry["frame"]
            frames.update(range(frame - N_IN, frame + N_OUT))
    return frames


def window_is_clean(start: int, person_mask: np.ndarray, forbidden: set) -> bool:
    if person_mask.shape[0] != WINDOW:
        raise ValueError(f"bad mask length {person_mask.shape[0]}")
    if not np.all(person_mask > 0.5):
        return False
    return not any(frame in forbidden for frame in range(start, start + WINDOW))


def build_windows_for_dataset(dataset, stepsize, test_json_path, max_windows=None):
    forbidden = forbidden_frames(dataset, test_json_path)
    scene = Scene.load_from_paths(
        dataset=dataset,
        person_path=poses_path(),
        scene_path=scenes_path(),
        smplx_path=dirname(smplx_path()),
    )
    # Slice windows directly from the Scene's raw per-frame arrays.
    # Do NOT use Scene.get_splits here: it densely materializes *every*
    # overlapping window (at stepsize 50 each frame lands in ~10 windows, and it
    # also expands smpls/transforms/activities we never use), which balloons to
    # tens of GB and OOMs even a 29GB box. Slicing the raw [n_frames, P, 29, 3]
    # array keeps peak memory at one Scene (~6GB) plus the kept clean windows.
    poses3d = scene.poses3d          # [n_frames, P, 29, 3], absolute-frame indexed
    masks = scene.masks              # [n_frames, P]
    starts = frame_splits(scene.frames, length=WINDOW, stepsize=stepsize)
    n_person = masks.shape[1]

    windows = []
    for start in starts:
        sl = slice(start, start + WINDOW)
        if masks[sl].shape[0] != WINDOW:
            continue
        for person_idx in range(n_person):
            if not window_is_clean(start, masks[sl, person_idx], forbidden):
                continue
            block = poses3d[sl, person_idx][:, None]   # [WINDOW, 1, 29, 3]
            try:
                normed, _ = normalize3d(block, frame=N_IN - 1)
            except ValueError:
                continue
            windows.append(normed[:, 0])
            if max_windows is not None and len(windows) >= max_windows:
                return np.asarray(windows, dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        window = self.windows[i].reshape(WINDOW, POSE_DIM)
        x = torch.tensor(window[:N_IN], dtype=torch.float32)
        y = torch.tensor(window[N_IN:], dtype=torch.float32)
        return x, y


class WholeWindowDataset(torch.utils.data.Dataset):
    """Yields the full normalized window as a diffusion target x0."""

    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        window = self.windows[i].reshape(WINDOW, POSE_DIM)
        return torch.tensor(window, dtype=torch.float32)


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


def build_or_load_windows(datasets, stepsize, test_json_path):
    tag = "".join(datasets) + f"_s{stepsize}"
    path = os.path.join(config.cache_dir(), f"windows_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    parts = [
        build_windows_for_dataset(dataset, stepsize, test_json_path)
        for dataset in datasets
    ]
    windows = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(path, windows)
    return windows


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
