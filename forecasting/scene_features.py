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
