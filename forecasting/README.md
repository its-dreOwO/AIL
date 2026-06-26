# Forecasting Baseline Runbook

This package contains the first HIK forecasting baseline: a single-person siMLPe-style
DCT-MLP model trained on canonicalized 500-frame windows and evaluated through the
vendored HIK `Evaluator.execute3d` protocol.

## Current Status

- Fast tests pass locally: `20 passed, 1 deselected`.
- Slow real-data window smoke test passes against dataset A.
- Zero-velocity baseline on dataset A:
  - overall mean MPJPE: `1.1080`
  - `@1s`: `0.5195`
  - `@5s`: `1.2535`
  - `@10s`: `1.4223`
- Local full training is not viable in this checkout: the process was killed with
  exit `137` during real-data cache/training startup.
- GPU VM created for remote training:
  - name: `hik-simlpe-train`
  - project: `project-b9a4f950-85a4-48f0-9ee`
  - zone: `asia-southeast1-b`
  - machine: `n1-standard-8`
  - GPU: `1 x nvidia-tesla-t4`
  - disk: `250GB pd-balanced`
  - external IP: `35.240.238.209`
  - `nvidia-smi` verified the Tesla T4 and NVIDIA driver.

## Dataset Path

The full dataset lives locally at:

```bash
/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen
```

Do not hardcode that path in code. Use `HIK_DATA`:

```bash
export HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen"
```

If uploaded to the VM with the command below, the remote path is:

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen/Humans_in_Kitchen"
```

## Upload Dataset To VM

Run this locally. It is verbose and should print transfer output while it runs.

```bash
CLOUDSDK_CONFIG=/tmp/gcloud-config gcloud compute scp \
  --recurse \
  --compress \
  --ssh-key-file=/tmp/google_compute_engine \
  "/mnt/elements/dataset AIL/Humans_in_Kitchen" \
  hik-simlpe-train:~/ \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --verbosity=info
```

Verify the upload on the VM:

```bash
CLOUDSDK_CONFIG=/tmp/gcloud-config gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --ssh-key-file=/tmp/google_compute_engine \
  --command='du -sh ~/Humans_in_Kitchen && find ~/Humans_in_Kitchen/Humans_in_Kitchen -maxdepth 1 -type d -print'
```

## Connect To VM

```bash
CLOUDSDK_CONFIG=/tmp/gcloud-config gcloud compute ssh hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b \
  --ssh-key-file=/tmp/google_compute_engine
```

## Remote Setup

On the VM:

```bash
git clone https://github.com/its-dreOwO/AIL.git hik
cd hik

# Use the feature branch once it has been pushed.
git fetch origin single-person-simlpe-baseline
git checkout single-person-simlpe-baseline

pip install -e .
pip install pytest
export HIK_DATA="$HOME/Humans_in_Kitchen/Humans_in_Kitchen"
```

Check the environment:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
python -m pytest tests/ -v -m "not slow"
python -m pytest tests/test_data.py -v -m slow
```

## Evaluate Zero Velocity

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen/Humans_in_Kitchen"
python -m forecasting.evaluate --dataset A --model zerovel
```

Local reference output for dataset A:

```text
overall mean MPJPE: 1.1080
  @1s: 0.5195
  @5s: 1.2535
  @10s: 1.4223
```

## Train siMLPe

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen/Humans_in_Kitchen"
python -m forecasting.train --datasets A B C D --stepsize 50 --epochs 80
```

Expected output:

- prints the number of materialized training windows
- prints one train/validation loss line per epoch
- writes `forecasting/cache/simlpe.pt`

## Evaluate siMLPe

```bash
export HIK_DATA="$HOME/Humans_in_Kitchen/Humans_in_Kitchen"
python -m forecasting.evaluate --dataset A --model simlpe --ckpt forecasting/cache/simlpe.pt
```

Record the overall mean and `@1s`, `@5s`, `@10s` values here after the remote run:

```text
siMLPe on A:
overall mean MPJPE: TBD
  @1s: TBD
  @5s: TBD
  @10s: TBD
```

## Stop Or Delete The VM

Stop the VM when not training:

```bash
CLOUDSDK_CONFIG=/tmp/gcloud-config gcloud compute instances stop hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b
```

Delete it when finished:

```bash
CLOUDSDK_CONFIG=/tmp/gcloud-config gcloud compute instances delete hik-simlpe-train \
  --project=project-b9a4f950-85a4-48f0-9ee \
  --zone=asia-southeast1-b
```
