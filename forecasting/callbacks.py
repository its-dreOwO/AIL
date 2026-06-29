import numpy as np
import torch

from hik.transforms.utils import backfill_masked, denormalize3d, normalize3d
from forecasting.config import N_IN, POSE_DIM
from forecasting.scene_features import extract_scene_features


def zero_velocity_callback(inp: dict) -> np.ndarray:
    poses_in = inp["Poses3d_in"]
    n_out = inp["n_out"]
    last = poses_in[-1]
    return np.repeat(last[None], n_out, axis=0).astype(np.float32)


def make_simlpe_callback(model, device="cpu"):
    model = model.to(device)
    model.eval()

    def callback(inp: dict) -> np.ndarray:
        poses_in = np.copy(inp["Poses3d_in"])
        masks_in = np.copy(inp["Masks_in"])
        n_out = inp["n_out"]
        n_person = poses_in.shape[1]
        fallback = zero_velocity_callback(inp)
        pred_world = np.empty((n_out, n_person, 29, 3), dtype=np.float32)

        filled, _ = backfill_masked(poses_in, masks_in)
        for pid in range(n_person):
            block = filled[:, pid : pid + 1]
            try:
                normed, norm_params = normalize3d(block, frame=N_IN - 1)
            except ValueError:
                pred_world[:, pid] = fallback[:, pid]
                continue

            x = normed[:, 0].reshape(1, N_IN, POSE_DIM)
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32, device=device)
                yt = model(xt)
            pred_normed = yt.cpu().numpy().reshape(n_out, 1, 29, 3)
            pred_world[:, pid : pid + 1] = denormalize3d(pred_normed, norm_params)
        return pred_world.astype(np.float32)

    return callback


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
                scene = extract_scene_features(
                    kitchen, frame=frame, pose3d=poses_in[-1, pid]
                )
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
