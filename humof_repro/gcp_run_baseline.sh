#!/bin/bash
# Preprocess + verify filters + launch the HUMOF baseline re-run (GATED_FFN=0) on the VM.
#
# Run ON the VM, after gcp_setup_baseline.sh and the dataset rsync have finished.
# Gate on GPU spend (per the design spec): the reconstructed filters must admit
# exactly 13,162 test cases on D. If they don't, the comparison to the original
# 109.71 is invalid at the root and no 24h of GPU time should be burned.
set -eux

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate humof
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=7.5
cd "$HOME/HUMOF"

# tus.pkl — skips SAST entirely (the repo's preprocess.py is broken: it windows
# the sequences and then asserts len(tus)==1).
if [ ! -f "$HOME/hik_preprocessed/H25F50/hik_D/tus.pkl" ]; then
  python "$HOME/hik/humof_repro/preprocess_hik_direct.py" \
    --dataset-root "$HOME/hik_dataset" \
    --out-dir "$HOME/hik_preprocessed/H25F50"
fi

python - <<'PY'
from datasets.dataset_hik import DatasetHik
n = len(DatasetHik(mode="test"))
print(f"len(dataset)={n}  target=13162  MATCH={n==13162}")
assert n == 13162, f"filter reconstruction drifted: {n} != 13162 — do not train"
PY

# Baseline arm: identical code to the gated arm, differing only in this flag.
export HUMOF_GATED_FFN=0
export HUMOF_RUN_TAG=base
export HUMOF_GPU_IDX=0
export HUMOF_NUM_WORKERS=8
setsid python main.py > "$HOME/train_base.log" 2>&1 &
echo "baseline training launched, pid $!, log ~/train_base.log"
