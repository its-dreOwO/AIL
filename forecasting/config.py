import os
from os.path import abspath, dirname, isdir, join

N_IN = 250
N_OUT = 250
WINDOW = N_IN + N_OUT
N_JOINTS = 29
POSE_DIM = N_JOINTS * 3
FPS = 25
DATASETS = ("A", "B", "C", "D")

DEFAULT_HIK_DATA = "/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen"

_REPO_ROOT = dirname(dirname(abspath(__file__)))


def data_root() -> str:
    return os.environ.get("HIK_DATA", DEFAULT_HIK_DATA)


def poses_path() -> str:
    return join(data_root(), "poses")


def scenes_path() -> str:
    return join(data_root(), "scenes")


def smplx_path() -> str:
    return join(data_root(), "body_models", "SMPLX_NEUTRAL.npz")


def cache_dir() -> str:
    d = join(_REPO_ROOT, "forecasting", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def dataset_available() -> bool:
    return isdir(poses_path())
