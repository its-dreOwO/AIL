#!/bin/bash
# Provision a fresh GCP T4 VM to run the HUMOF baseline re-run (GATED_FFN=0).
#
# Run ON the VM. Rebuilds the env the deleted VM had, per the recipe in CLAUDE.md.
# The pins are not arbitrary:
#   - torch 1.13.0+cu117: HUMOF's requirements.txt pins torch==1.12.0+cu117, a
#     wheel that never existed; its tv0.14.0/ta0.13.0 pins imply 1.13.0.
#   - nvcc 11.7 via conda cuda-toolkit: pvcnn JIT-compiles CUDA at import.
#   - ln -sfn $CONDA_PREFIX/lib $CONDA_PREFIX/lib64: pvcnn's build otherwise
#     fails with -lcudart not found.
#   - TORCH_CUDA_ARCH_LIST=7.5: T4 is sm_75.
set -eux

MC=$HOME/miniconda3
if [ ! -d "$MC" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC"
fi
source "$MC/etc/profile.d/conda.sh"

# conda-forge only, --override-channels: Anaconda's `defaults` channel now refuses
# non-interactive use until its Terms of Service are accepted (CondaToSNonInteractiveError),
# which did not apply when the original VM was built. conda-forge has no such gate and
# gives an equivalent python 3.9 — torch/numpy come from pip wheels, so numerics are unaffected.
conda create -y -n humof python=3.9 --override-channels -c conda-forge
conda activate humof

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.24.3 ninja tqdm natsort pandas pytz tensorboardX 'protobuf<4' \
    torchgeometry numba einops matplotlib scipy scikit-learn smplx opencv-python-headless

conda install -y --override-channels -c nvidia/label/cuda-11.7.0 -c conda-forge cuda-toolkit
# pvcnn's JIT looks for lib64; conda only ships lib.
ln -sfn "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"

# pvcnn JIT-compiles CUDA at import, so it needs a host compiler. The GCP
# deeplearning image ships gcc-12 and NO g++/c++ at all, which breaks two ways:
# torch's cpp_extension aborts on `which c++`, and nvcc 11.7 refuses gcc > 11
# ("unsupported GNU version"). Install 11 and point the env's PATH at it —
# $CONDA_PREFIX/bin precedes /usr/bin, so these win over the system gcc-12.
sudo apt-get update -qq
sudo apt-get install -y -qq gcc-11 g++-11
ln -sf /usr/bin/gcc-11 "$CONDA_PREFIX/bin/gcc"
ln -sf /usr/bin/g++-11 "$CONDA_PREFIX/bin/g++"
ln -sf /usr/bin/g++-11 "$CONDA_PREFIX/bin/c++"

# hik data API (vendored copy — the commit requirements.txt pins is gone upstream)
pip install --no-deps -e "$HOME/hik"

cd "$HOME/HUMOF"
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=7.5

# Wire HUMOF's hardcoded relative paths at the dataset. This comes BEFORE the
# pvcnn check on purpose: with `set -e`, a failed *verification* used to abort the
# script and silently skip this *setup*, leaving dangling paths that surface later
# as a confusing FileNotFoundError from deep inside preprocess.
mkdir -p datasets/dataset_preprocess/hik/SAST
ln -sfn "$HOME/hik_dataset/scenes" datasets/dataset_preprocess/hik/SAST/scenes
mkdir -p datasets/dataset_preprocess/hik/hik_preprocessed
ln -sfn "$HOME/hik_preprocessed/H25F50" datasets/dataset_preprocess/hik/hik_preprocessed/H25F50
mkdir -p "$HOME/hik_preprocessed/H25F50/tu_2_others_ids"
mkdir -p "$HOME/hik_preprocessed/H25F50/tu_2_should_filter_primary"

# Verify last: pvcnn JIT-compiles CUDA at import, so this is the real smoke test.
python -c 'from pvcnn.modules.functional.backend import _backend; print("pvcnn OK")'

echo "SETUP OK — next: preprocess, then HUMOF_GATED_FFN=0 python main.py"
