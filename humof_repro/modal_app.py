"""Modal app for HUMOF HIK training — Phase 2 gated-FFN experiment.

Why Modal instead of the GCP T4 VM:
  The GCP project's GPUS_ALL_REGIONS quota is 1, so the gated arm and the
  baseline re-run cannot run concurrently there. The baseline re-run is what
  measures the GPU-nondeterminism noise floor (pvcnn's voxelization uses
  atomicAdd scatters, which are nondeterministic and compound over 70 epochs).
  Without that floor, the gated-vs-baseline delta is not interpretable.
  Two parallel A100 arms give us both numbers in one wall-clock window.

Hardware note: torch 1.13.0+cu117 supports up to sm_86, so H100 (sm_90) is not
usable without changing torch — which would change numerics. A100 (sm_80) is
the ceiling. Both arms run on the same GPU type so hardware cancels out.

Usage:
    modal run humof_repro/modal_app.py::smoke        # verify image + pvcnn JIT
"""

import modal

APP_NAME = "humof-hik"

# HUMOF's requirements.txt pins hik to 438b4fe..., but that commit no longer
# exists upstream (git fetch -> "upload-pack: not our ref"); jutanke/hik has
# rewritten history since publication. We install our vendored copy instead,
# which is what the deleted VM used too (`pip install -e ~/hik`).
HIK_LOCAL = "/opt/study/ail/hik"
HUMOF_LOCAL = "/opt/study/ail/HUMOF"

app = modal.App(APP_NAME)

# Dataset + preprocessed pickles + checkpoints live here so they survive across
# containers. (The last VM's disk was deleted and took the env, the dataset, the
# preprocessed pickles and all of Phase 1's code with it. Not again.)
data_vol = modal.Volume.from_name("humof-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("humof-ckpt", create_if_missing=True)

image = (
    # 11.7.1-devel gives us nvcc 11.7, which pvcnn's JIT needs.
    # py3.10 (not the VM's 3.9 — Modal dropped <3.10); torch 1.13.0+cu117 ships
    # cp310 wheels and the numerics come from torch/CUDA, not the python minor.
    modal.Image.from_registry(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==1.13.0+cu117",
        "torchvision==0.14.0+cu117",
        extra_index_url="https://download.pytorch.org/whl/cu117",
    )
    .pip_install(
        "numpy==1.24.3",
        "ninja",
        "tqdm",
        "natsort",
        "pandas",
        "pytz",
        "tensorboardX",
        "protobuf<4",
        "torchgeometry",  # datasets/aug.py
        # hik package deps
        "numba",
        "einops",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "smplx",
        "opencv-python-headless",
    )
    # Vendored hik package (388KB): just the package dir + setup.py.
    .add_local_dir(f"{HIK_LOCAL}/hik", "/root/hikpkg/hik", copy=True,
                   ignore=["__pycache__/**", "**/__pycache__/**"])
    .add_local_file(f"{HIK_LOCAL}/setup.py", "/root/hikpkg/setup.py", copy=True)
    .run_commands("pip install --no-deps -e /root/hikpkg")
    # Our authored scripts (preprocess, filter reconstruction).
    .add_local_dir(f"{HIK_LOCAL}/humof_repro", "/root/humof_repro", copy=True,
                   ignore=["__pycache__/**"])
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0",  # A100 = sm_80
            "TORCH_EXTENSIONS_DIR": "/root/torch_ext",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # 960KB of code once the log/checkpoints/pdf are excluded.
    .add_local_dir(
        HUMOF_LOCAL,
        "/root/HUMOF",
        copy=True,
        ignore=[
            "*.log",
            "checkpoints/**",
            "*.pdf",
            ".git/**",
            "data/**",
            "results/**",
            "__pycache__/**",
            "**/__pycache__/**",
        ],
    )
    # Bake the pvcnn CUDA build into the image so containers don't each pay the
    # ~5min JIT cost (and so a compile failure surfaces at build, not at hour 1
    # of training). No GPU needed to compile — nvcc targets sm_80 via
    # TORCH_CUDA_ARCH_LIST.
    .run_commands(
        "cd /root/HUMOF && python -c 'from pvcnn.modules.functional.backend import _backend; print(\"pvcnn backend OK\")'",
        gpu="A100",
    )
)


def _setup_paths():
    """Point HUMOF's relative data paths at the Modal volumes.

    conf.py hardcodes ./datasets/dataset_preprocess/hik/{SAST,hik_preprocessed}
    relative to cwd, so we chdir into the repo and symlink those at the volume.
    """
    import os
    import sys

    os.chdir("/root/HUMOF")
    sys.path.insert(0, "/root/HUMOF")

    base = "datasets/dataset_preprocess/hik"
    os.makedirs(f"{base}/SAST", exist_ok=True)
    os.makedirs(f"{base}/hik_preprocessed", exist_ok=True)
    os.makedirs("/data/hik_preprocessed/H25F50", exist_ok=True)
    # DatasetHik writes its filter caches into these two.
    os.makedirs("/data/hik_preprocessed/H25F50/tu_2_others_ids", exist_ok=True)
    os.makedirs("/data/hik_preprocessed/H25F50/tu_2_should_filter_primary", exist_ok=True)

    for link, target in [
        (f"{base}/SAST/scenes", "/data/hik_dataset/scenes"),
        (f"{base}/hik_preprocessed/H25F50", "/data/hik_preprocessed/H25F50"),
    ]:
        if os.path.islink(link):
            os.unlink(link)
        if not os.path.exists(link):
            os.symlink(target, link)


@app.function(
    image=image, cpu=8, memory=32768, timeout=7200,
    volumes={"/data": data_vol},
)
def preprocess(kids: str = "ABCD"):
    """Build tus.pkl for each recording straight from raw HIK (bypasses SAST)."""
    _setup_paths()
    import sys

    sys.path.insert(0, "/root/humof_repro")
    from preprocess_hik_direct import build

    build("/data/hik_dataset", "/data/hik_preprocessed/H25F50", kids)
    data_vol.commit()
    print("PREPROCESS DONE")


@app.function(
    image=image, gpu="A100", cpu=8, memory=32768, timeout=7200,
    volumes={"/data": data_vol},
)
def verify_filters():
    """Gate on GPU spend: does the re-reconstructed filter match Phase 0?

    Targets (recording D, mode='test'): 69,290 absent / 284 partial /
    21,186 rejected by primary_filterC / 22 no-others / 13,162 admitted.
    """
    _setup_paths()
    import os

    os.environ["HUMOF_GATED_FFN"] = "0"

    from datasets.dataset_hik import DatasetHik

    ds = DatasetHik(mode="test")
    n = len(ds)
    print(f"\n{'='*50}\nlen(dataset) = {n}   (Phase 0 target: 13162)")
    print(f"MATCH: {n == 13162}\n{'='*50}")
    data_vol.commit()
    return n


@app.function(
    image=image, gpu="A100", cpu=8, memory=32768,
    timeout=24 * 3600,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
)
def train(gated: bool = True, tag: str = None, epochs: int = 70, save_interval: int = 2):
    """Run HUMOF training. Detach this — it takes ~9-14h on an A100.

    Restart-safe: Modal preempts containers and re-runs the function from the
    top (observed once, ~8 min in), which would otherwise restart training at
    epoch 0 forever. We resume from the newest checkpoint on the volume instead.
    """
    _setup_paths()
    import os
    import subprocess

    tag = tag or ("gated" if gated else "base")
    env = dict(os.environ)
    env["HUMOF_GATED_FFN"] = "1" if gated else "0"
    env["HUMOF_RUN_TAG"] = tag
    env["HUMOF_GPU_IDX"] = "0"
    env["HUMOF_NUM_WORKERS"] = "8"
    env["HUMOF_SAVE_INTERVAL"] = str(save_interval)

    # Persist checkpoints/results on the volume, not the ephemeral container.
    for local, remote in [("checkpoints", f"/ckpt/{tag}/checkpoints"),
                          ("results", f"/ckpt/{tag}/results")]:
        os.makedirs(remote, exist_ok=True)
        if os.path.islink(local):
            os.unlink(local)
        elif os.path.isdir(local):
            import shutil

            shutil.rmtree(local)
        os.symlink(remote, local)

    # Resume from the newest checkpoint if one exists. main.py already restores
    # model/optimizer/scheduler and runs range(cp_iter, num_epoch) when given an
    # integer HUMOF_CP_ITER; leaving it unset starts fresh. We autodetect here
    # rather than using main.py's 'auto' branch, which int()s the env var before
    # it can match 'auto' and then blocks on input() anyway.
    import glob

    ckpt_vol.reload()  # a prior container's commits are not visible until reload
    ck_dir = f"/ckpt/{tag}/checkpoints/hik-releaseV0.1-{tag}"
    done = [int(os.path.basename(f)[:-4]) for f in glob.glob(f"{ck_dir}/*.pth")]
    if done:
        env["HUMOF_CP_ITER"] = str(max(done))
        print(f"=== resuming from epoch {max(done)} (found {sorted(done)}) ===", flush=True)
    else:
        print("=== no checkpoint found, training from scratch ===", flush=True)

    # Modal volumes only persist on an explicit commit(), so a crash at hour 8
    # of a 9h run would lose every checkpoint. Commit periodically instead.
    import threading

    stop = threading.Event()

    def _committer():
        while not stop.wait(600):
            try:
                ckpt_vol.commit()
                print("[committer] volume committed", flush=True)
            except Exception as e:  # never let this kill the run
                print(f"[committer] commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    print(f"=== training arm={tag} gated={gated} epochs={epochs} ===", flush=True)
    try:
        p = subprocess.run(["python", "main.py"], cwd="/root/HUMOF", env=env)
    finally:
        stop.set()
        ckpt_vol.commit()
    print(f"exit={p.returncode}")
    return p.returncode


@app.function(
    image=image, gpu="A100", cpu=8, memory=32768, timeout=3600,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
)
def eval_ckpt(cp_iter: int = 70, gated: bool = False, tag: str = "evalref"):
    """Eval an existing checkpoint through the rebuilt pipeline.

    This is the end-to-end check the static verifications (filter counts, param
    count) cannot give: they prove the data subset and the architecture match,
    not that the pipeline reproduces the same *number*. Evaluating the surviving
    70.pth here answers two things at once:
      1. does the rebuilt eval path reproduce ~109.71 (i.e. is 109.71 still a
         valid reference)?
      2. what is the eval-side T4->A100 gap, measured rather than hand-waved?
    """
    _setup_paths()
    import os
    import subprocess

    env = dict(os.environ)
    env["HUMOF_GATED_FFN"] = "1" if gated else "0"
    env["HUMOF_RUN_TAG"] = tag
    env["HUMOF_TRAIN"] = "0"          # test mode -> single eval pass
    env["HUMOF_CP_ITER"] = str(cp_iter)
    env["HUMOF_GPU_IDX"] = "0"
    env["HUMOF_NUM_WORKERS"] = "8"

    for local, remote in [("checkpoints", f"/ckpt/{tag}/checkpoints"),
                          ("results", f"/ckpt/{tag}/results")]:
        os.makedirs(remote, exist_ok=True)
        if os.path.islink(local):
            os.unlink(local)
        elif os.path.isdir(local):
            import shutil

            shutil.rmtree(local)
        os.symlink(remote, local)

    p = subprocess.run(["python", "main.py"], cwd="/root/HUMOF", env=env)
    ckpt_vol.commit()

    err = f"/ckpt/{tag}/results/hik-releaseV0.1-{tag}/err.csv"
    if os.path.exists(err):
        print(f"\n===== {err} =====")
        print(open(err).read())
    return p.returncode


@app.function(image=image, gpu="A100", timeout=900, volumes={"/data": data_vol})
def smoke():
    """Verify the image: torch, CUDA, pvcnn extension, hik import."""
    import subprocess
    import sys

    sys.path.insert(0, "/root/HUMOF")

    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("nvcc:", subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.split("release")[-1].strip()[:20])

    from pvcnn.modules.functional.backend import _backend

    print("pvcnn backend:", _backend)

    import hik
    from hik.data.scene import Scene

    print("hik OK:", hik.__file__)

    # Confirm TF32 can be pinned off — Ampere enables cuDNN TF32 by default,
    # which would differ numerically from the T4-trained baseline.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print("tf32 cudnn:", torch.backends.cudnn.allow_tf32,
          "| tf32 matmul:", torch.backends.cuda.matmul.allow_tf32)

    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok:", (x @ x).sum().item())
    print("SMOKE OK")
