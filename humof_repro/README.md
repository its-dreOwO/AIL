# humof_repro — rebuilding the HUMOF HIK pipeline

Everything needed to reproduce our HUMOF-on-HIK runs, **kept in this repo on purpose**.

## Why this directory exists

On 2026-07-15 the GCP VM `hik-simlpe-train` was deleted with no snapshot and no
orphaned disk. Everything that lived only on that VM was lost:

- the conda env (torch 1.13.0+cu117, JIT-compiled pvcnn)
- the dataset copy and the preprocessed `tus.pkl` files
- `preprocess_hik_direct.py`
- **the reconstructed `primary_filterC` / `filterB`** — the functions the whole
  eval subset depends on
- all of Phase 1's code (TTA hook, `worker_init_fn` determinism fix, `avg_ckpt.py`)
- checkpoints 10–50

What survived did so only because it happened to be copied into this repo
(`70.pth`, `60.pth`, the Phase 0/1 result CSVs and write-ups). The write-ups are
what made recovery possible at all: `docs/results/humof-improve/subset-report.md`
recorded the filters' semantics *and* their exact expected counts, which is the
only reason the reconstruction is verifiable rather than guesswork.

**Rule going forward: nothing that matters lives only on compute.** Code goes
here; compute is disposable.

## Contents

| File | What it is |
|---|---|
| `modal_app.py` | Modal app: image (CUDA 11.7 + torch 1.13.0+cu117 + pvcnn), volumes, `preprocess` / `verify_filters` / `train` / `eval_ckpt` / `smoke` functions |
| `preprocess_hik_direct.py` | Builds `hik_preprocessed/H25F50/hik_{A..D}/tus.pkl` from raw HIK, bypassing the broken `preprocess.py` and SAST entirely. Needs `poses/`, `scenes/` **and `body_models/`** (`Scene` constructs a SMPL-X `Body`). |
| `humof_addons.py` | Reconstructed `primary_filterC` / `filterB` + their verification targets |
| `patches/phase2-gated-ffn.patch` | All edits to the sibling HUMOF checkout (SwiGLU, config fixes, TF32 pin, worker seeding, env-driven save interval) |
| `gcp_grab_t4.sh` | Cycles all 51 T4 zones until one has capacity, then creates the VM |
| `gcp_setup_baseline.sh` | Provisions a fresh GCP T4 (conda env, torch, cuda-toolkit, pvcnn JIT, symlinks) |
| `gcp_run_baseline.sh` | On-VM: preprocess → gate on filter count → launch the baseline arm detached |

## Bugs found in the HUMOF repo as shipped

These are real and worth reporting in the write-up:

1. **`primary_filterC` / `filterB` were never committed.** Referenced by
   `datasets/dataset_hik.py:148,233`; absent from HUMOF's git history and from
   SAST. Reconstructed in `humof_addons.py`.
2. **`conf0.py` ends with `DATASET_name='humanise'`** — the last assignment wins,
   so the repo as-shipped does not run HIK.
3. **`conf2.py` sets `CUDA_VISIBLE_DEVICES='1'`** — selects a second GPU, so on a
   1-GPU machine it finds no CUDA device.
4. **`preprocess.py` is broken** — windows the recording, then asserts
   `len(tus)==1`, which cannot hold.
5. **`requirements.txt` pins `torch==1.12.0+cu117`**, a wheel that never existed;
   its own torchvision/torchaudio pins imply 1.13.0.
6. **`requirements.txt` pins `hik` to commit `438b4fe…`, which no longer exists
   upstream** (`git fetch` → `upload-pack: not our ref`).
7. **`get_others` has a latent infinite loop**: the `while 1:` retry only widens
   `max_dist_from_primary_to_other`, which is ignored when `ENABLE_filterB=1`.
   Only terminates because `MIN_OTHER_NUM=0`.
8. **`main.py`'s `'auto'` checkpoint-resume branch cannot fire.** Line 7 does
   `int(os.environ.get('HUMOF_CP_ITER','0'))`, which raises on `'auto'` long
   before the `cp_iter=='auto'` test at line 458 — and if it did fire, line 461
   blocks on `input()`, which hangs headless. Pass a real integer instead;
   `modal_app.py::train` autodetects it.

## Environment drift (things that changed since the original VM was built)

- **conda's `defaults` channel now refuses non-interactive installs**
  (`CondaToSNonInteractiveError`) until its Terms of Service are accepted.
  `gcp_setup_baseline.sh` pins `--override-channels -c conda-forge` to sidestep
  it; torch/numpy come from pip wheels, so numerics are unaffected.
- **Modal preempts containers** and re-runs the function from the top. Observed
  ~8 min into a run: `KeyboardInterrupt` inside the dataloader poll, followed by
  `Runner terminated` — an external kill, not a data bug. Without resume this
  restarts training at epoch 0 indefinitely.
- **Modal Volumes only persist on an explicit `commit()`** — a crash at hour 8 of
  a 9h run loses every checkpoint. `train()` runs a periodic committer thread.
- **`modal app stop` needs `-y`** (it aborts with "no interactive terminal
  detected" otherwise). A silently-failed stop once left two A100s running
  concurrently at $4.20/hr. Always verify with `modal app list` after launching.

## Reproducing

```bash
# 1. dataset -> Modal volume (3.9GB, complete: poses/ scenes/ body_models/)
modal volume put humof-data /home/dre/Downloads/hik_dataset/data /hik_dataset

# 2. build tus.pkl for A-D
modal run --detach humof_repro/modal_app.py::preprocess

# 3. GATE: filters must reproduce Phase 0's count before spending GPU hours
modal run humof_repro/modal_app.py::verify_filters     # must print len(dataset)=13162

# 4. train (detached — ~7.5 min/epoch on A100, ~9h for 70 epochs)
#    NOTE: modal treats bools as flags — `--gated=True` fails with
#    "Option '--gated' does not take a value".
modal run --detach humof_repro/modal_app.py::train --gated --tag gated
modal run --detach humof_repro/modal_app.py::train         --tag base

# 5. eval an existing checkpoint through the rebuilt pipeline (~15 min)
modal run --detach humof_repro/modal_app.py::eval_ckpt --cp-iter 70 --tag evalref
```

Training is **restart-safe**: `train()` globs the newest `*.pth` on the volume and
sets `HUMOF_CP_ITER`, so a preempted run resumes (model + optimizer + scheduler)
instead of restarting at epoch 0. `HUMOF_SAVE_INTERVAL=2` bounds the loss to ~24
min; multiples of 10 are still saved, so Phase 1's checkpoint-averaging set stays
a subset. Caveat: resume does **not** restore the global torch RNG, so a
restarted run's dropout/aug draws diverge from an uninterrupted one — one more
entry on the confound list, inside the noise band.

For the GCP T4 arm instead of Modal:

```bash
bash humof_repro/gcp_grab_t4.sh                 # acquire a T4 (capacity is scarce)
# then, on the VM:
bash ~/gcp_setup_baseline.sh && bash ~/gcp_run_baseline.sh
```

To apply the HUMOF-side edits to a fresh clone:

```bash
git clone https://github.com/scy639/HUMOF /opt/study/ail/HUMOF
cd /opt/study/ail/HUMOF && git apply /opt/study/ail/hik/humof_repro/patches/phase2-gated-ffn.patch
cp /opt/study/ail/hik/humof_repro/humof_addons.py .
```
