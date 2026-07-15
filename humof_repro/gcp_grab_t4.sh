#!/bin/bash
# Poll every T4 zone until one has capacity, then create the VM and stop.
#
# Context: GPUS_ALL_REGIONS quota is 1 and every region's NVIDIA_T4_GPUS quota is
# 1, so exactly one T4 anywhere is allowed — the blocker is capacity, not quota.
# Credit/trial projects get low priority for GPU allocation, so all 51 zones can
# be simultaneously out of stock; capacity frees up unpredictably.
#
# Exits 0 on success, writing the winning zone to gcp_zone.txt.

set -u
P=project-b9a4f950-85a4-48f0-9ee
NAME=hik-humof-base
OUT_ZONE="${1:-/tmp/gcp_zone.txt}"
LOG="${2:-/tmp/gcp_grab.log}"

ZONES="asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-southeast2-a asia-southeast2-b
asia-east1-a asia-east1-c asia-east2-a asia-east2-c
asia-northeast1-a asia-northeast1-c asia-northeast3-b asia-northeast3-c
asia-south1-a asia-south1-b
us-central1-a us-central1-b us-central1-c us-central1-f
us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c
us-west1-a us-west1-b us-west2-b us-west2-c us-west3-b us-west4-a us-west4-b
europe-west1-b europe-west1-c europe-west1-d europe-west2-a europe-west2-b
europe-west3-b europe-west4-a europe-west4-b europe-west4-c
europe-central2-b europe-central2-c
northamerica-northeast1-c southamerica-east1-a southamerica-east1-b southamerica-east1-c
australia-southeast1-a australia-southeast1-c me-west1-b me-west1-c"

ROUND=0
while true; do
  ROUND=$((ROUND + 1))
  echo "=== round $ROUND $(date '+%H:%M:%S') ===" >> "$LOG"
  for Z in $ZONES; do
    OUT=$(timeout 180 gcloud compute instances create "$NAME" \
      --project=$P --zone="$Z" --machine-type=n1-standard-8 \
      --accelerator=type=nvidia-tesla-t4,count=1 \
      --maintenance-policy=TERMINATE \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=150GB --boot-disk-type=pd-balanced \
      --metadata="install-nvidia-driver=True" 2>&1)

    if echo "$OUT" | grep -qE "RUNNING|STAGING"; then
      echo "$Z" > "$OUT_ZONE"
      echo "SUCCESS zone=$Z round=$ROUND $(date '+%F %H:%M:%S')" | tee -a "$LOG"
      exit 0
    fi

    # Already exists => a previous round won; treat as success.
    if echo "$OUT" | grep -q "already exists"; then
      echo "$Z" > "$OUT_ZONE"
      echo "ALREADY EXISTS zone=$Z" | tee -a "$LOG"
      exit 0
    fi

    # Quota errors are permanent for this config; capacity errors are transient.
    if echo "$OUT" | tr '\n' ' ' | grep -qE "Quota .* exceeded|QUOTA_EXCEEDED"; then
      echo "$Z: QUOTA (permanent)" >> "$LOG"
    else
      echo "$Z: no capacity" >> "$LOG"
    fi
  done
  echo "round $ROUND: all zones out of stock, sleeping 90s" >> "$LOG"
  sleep 90
done
