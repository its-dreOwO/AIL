# HUMOF Baseline Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve our HUMOF HIK recording-D subset MPJPE with zero-retrain-confound levers (checkpoint averaging, test-time rotation averaging), and characterize the reconstructed-filter eval subset as a reproducibility finding.

**Architecture:** All work runs on the GCP T4 VM (`ssh hikvm`) against `~/HUMOF` (the runnable copy with reconstructed filters + working `humof` conda env). The local clone `/opt/study/ail/HUMOF` is partial (only results/checkpoints/log were pulled) — do NOT trust its `conf.py` for exact knobs. Eval is invoked by setting `TRAIN=False` + `cp_iter=N` and running `python main.py`, which loads `checkpoints/hik-releaseV0.1/N.pth`, builds the test dataset (recording D), and writes `results/hik-releaseV0.1/err.csv`. Test eval is deterministic (`seed=0`, `SHUFFLE_when_test=False`).

**Tech Stack:** PyTorch 1.13.0+cu117, HUMOF repo, HIK harness (`hik`), conda env `humof`, GCP T4.

**Experiment-plan note:** This is remote experiment orchestration, not local TDD. "Verify it fails/passes" is replaced by "run and confirm the expected printed output / CSV row." Every run produces a durable artifact (CSV / saved checkpoint / log) — commit or pull those.

---

## File Structure

On the VM (`~/HUMOF`):
- `conf.py` / `conf0.py` / `conf2.py` — config knobs: `TRAIN`, `cp_iter`, `model_path_trained`, filter thresholds. **Exact locations discovered in Task 0.**
- `main.py` — driver + `train(epoch, TRAIN, dataset)` eval loop (predictions `y` at the IDCT step, error accumulated right after).
- `datasets/dataset_hik.py` — builds recording-D windows; already prints `__ct_filtered_1/2` + `__ct_not_filtered` counters (lines ~113–160).
- `datasets/aug.py::A` — z-axis rotation aug (reused for TTA convention).
- New: `~/HUMOF/tools/avg_ckpt.py` — weight-average checkpoints → one `.pth`.
- New: `~/HUMOF/tools/subset_report.py` — Phase-0 subset counts (or inline instrumentation).

Deliverables pulled to local `/opt/study/ail/HUMOF/results/` + written up under `hik/docs/`.

---

## Task 0: Bring up VM, snapshot exact config knobs

**Files:** none created; read-only discovery.

- [ ] **Step 1: Refresh VM IP and connect**

```bash
gcloud compute instances start hik-simlpe-train --zone=asia-southeast1-b
IP=$(gcloud compute instances describe hik-simlpe-train --zone=asia-southeast1-b --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
# update HostName for `hikvm` in ~/.ssh/config to $IP (per CLAUDE.md), then:
ssh hikvm 'echo connected'
```
Expected: `connected`.

- [ ] **Step 2: Snapshot the eval knobs (do NOT guess from the local clone)**

```bash
ssh hikvm 'cd ~/HUMOF && grep -rn "TRAIN *=\|cp_iter\|model_path_trained\|SHUFFLE_when_test\|THRES__primary_filterC\|THRES_filterB\|STEP_test\|PRIMARY_FILTER_C\|ENABLE_filterB" conf.py conf0.py conf2.py main.py'
ssh hikvm 'ls -la ~/HUMOF/checkpoints/hik-releaseV0.1/'
```
Expected: prints the file:line of `TRAIN`, `cp_iter`, `model_path_trained`, and the filter thresholds; checkpoint listing shows `10.pth … 70.pth`. **Record these exact locations — later tasks reference "the `cp_iter` line" etc.**

- [ ] **Step 3: Reproduce the single-checkpoint baseline (sanity that eval still runs)**

Set `TRAIN=False` and `cp_iter=70` (at the locations from Step 2; `cp_iter` prompts `y/n` on AUTO — set a literal `70` to avoid the prompt), then:
```bash
ssh hikvm 'cd ~/HUMOF && conda run -n humof env TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=$CONDA_PREFIX python main.py 2>&1 | tail -20'
ssh hikvm 'cat ~/HUMOF/results/hik-releaseV0.1/err.csv | tail -4'
```
Expected: `path_err … mean ≈ 109.8`, `joint_err … mean ≈ 66.8` (matches the committed baseline). This is the number every later phase A/Bs against.

- [ ] **Step 4: Commit the baseline snapshot locally**

```bash
# from local hik repo, after pulling the confirmed err.csv into a phase-0 dir
mkdir -p /opt/study/ail/hik/docs/results/humof-improve
scp hikvm:~/HUMOF/results/hik-releaseV0.1/err.csv /opt/study/ail/hik/docs/results/humof-improve/baseline-ckpt70.csv
cd /opt/study/ail/hik && git add docs/results/humof-improve/baseline-ckpt70.csv && git commit -m "chore(humof): snapshot single-ckpt70 baseline (path 109.8 / pose 66.8)"
```

---

## Task 1: Phase 0 — subset characterization (finding)

**Files:**
- Modify (VM): `~/HUMOF/datasets/dataset_hik.py` (extend existing counters ~lines 113–160)
- Create (local): `hik/docs/results/humof-improve/subset-report.md`

- [ ] **Step 1: Add two missing rejection counters + total, and a displacement dump**

In `dataset_hik.py`, the loop already counts `__ct_filtered_1` (absent), `__ct_filtered_2` (partial presence), `__ct_not_filtered` (admitted). Add counters for the two currently-silent rejections and record admitted vs. raw. Edit the loop body:

```python
# near the other counters (~line 115)
__ct_filtered_primary = 0     # rejected by primary_filterC
__ct_filtered_noother = 0     # rejected by len_others == 0
__ct_candidates = 0           # present+full windows that reach the filters
```
```python
# where should_filter_primary is True (~line 150-151), replace `continue`:
                    if should_filter_primary:
                        __ct_filtered_primary += 1
                        continue
```
```python
# where len_others == 0 (~line 153-154), replace `continue`:
                if len_others == 0:
                    __ct_filtered_noother += 1
                    continue
```
```python
# just after `tu2 = (kid, iF, ip)` (~line 140), count every candidate that reaches the filters:
                    __ct_candidates += 1
```
```python
# alongside the existing prints (~line 158-160):
        print(f'{__ct_filtered_primary=}')
        print(f'{__ct_filtered_noother=}')
        print(f'{__ct_candidates=}')
        print(f'SUBSET_ADMIT_RATE={__ct_not_filtered}/{__ct_candidates}')
```

- [ ] **Step 2: Run the test-split construction and capture the counts**

```bash
ssh hikvm 'cd ~/HUMOF && conda run -n humof python -c "
import conf, globals_; globals_.TRAIN=False
from datasets.dataset_hik import DatasetHik
ds = DatasetHik(\"test\")
print(\"LEN\", len(ds))
" 2>&1 | grep -E "__ct_|SUBSET_ADMIT|LEN"'
```
Expected: prints admitted / candidate counts on recording D. (If `DatasetHik`/`conf` import names differ, adjust from the Task-0 Step-2 grep.)

- [ ] **Step 3: Characterize admitted vs. rejected by motion displacement**

`primary_filterC` thresholds on most-moving-joint bbox diagonal. Add a one-off dump of that statistic for admitted vs. primary-rejected windows (append to the loop, guarded by an env flag so it doesn't slow normal runs), OR compute it standalone from `self.kid__2__ip_2_iF2joints['D']`. Capture: mean/median displacement of admitted vs. rejected, and the admit rate.

- [ ] **Step 4: Write the finding**

Create `hik/docs/results/humof-improve/subset-report.md` with: admitted count, candidate count, admit rate, per-rejection-reason breakdown, and displacement comparison. State plainly whether the subset is materially smaller/easier than raw → the reproducibility limitation.

- [ ] **Step 5: Commit**

```bash
cd /opt/study/ail/hik && git add docs/results/humof-improve/subset-report.md && git commit -m "docs(humof): Phase 0 subset characterization finding"
# also commit the VM dataset_hik.py counter diff into the repo copy if tracked, else note it in the report
```

---

## Task 2: Phase 1a — checkpoint weight averaging

**Files:**
- Create (VM): `~/HUMOF/tools/avg_ckpt.py`
- Config: the `cp_iter` / `model_path_trained` lines from Task 0

- [ ] **Step 1: Write the averaging script**

```python
# ~/HUMOF/tools/avg_ckpt.py
import sys, torch
# usage: python tools/avg_ckpt.py OUT.pth CKPT1.pth CKPT2.pth ...
out, paths = sys.argv[1], sys.argv[2:]
assert len(paths) >= 2, "need >=2 checkpoints to average"
states = [torch.load(p, map_location="cpu", weights_only=True)["model_dict"] for p in paths]  # own checkpoints, tensors only
avg = {}
for k in states[0]:
    if states[0][k].is_floating_point():
        avg[k] = sum(s[k].float() for s in states) / len(states)
    else:
        avg[k] = states[0][k]  # ints (e.g. num_batches_tracked): copy, don't average
torch.save({"model_dict": avg}, out)
print(f"wrote {out} averaging {len(paths)} ckpts")
```

- [ ] **Step 2: Produce the averaged checkpoint (last 3: 50, 60, 70)**

```bash
ssh hikvm 'cd ~/HUMOF && conda run -n humof python tools/avg_ckpt.py checkpoints/hik-releaseV0.1/avg_50-60-70.pth checkpoints/hik-releaseV0.1/50.pth checkpoints/hik-releaseV0.1/60.pth checkpoints/hik-releaseV0.1/70.pth'
```
Expected: `wrote …/avg_50-60-70.pth averaging 3 ckpts`.

- [ ] **Step 3: Eval the averaged checkpoint**

The eval loader reads `torch.load(cp_path)['opt_dict'/'scheduler_dict'/'model_dict']`. The avg checkpoint has only `model_dict`. Point eval at it WITHOUT requiring optimizer/scheduler: temporarily guard the loader (in `main.py` ~line 454-455) so `opt_dict`/`scheduler_dict` load only `if 'opt_dict' in model_cp`. Then set `model_path_trained` (or a direct path override) to `avg_50-60-70.pth` and run:
```bash
ssh hikvm 'cd ~/HUMOF && conda run -n humof env TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=$CONDA_PREFIX python main.py 2>&1 | tail -6'
ssh hikvm 'cp ~/HUMOF/results/hik-releaseV0.1/err.csv ~/HUMOF/results/hik-releaseV0.1/err_avg_50-60-70.csv && tail -4 ~/HUMOF/results/hik-releaseV0.1/err_avg_50-60-70.csv'
```
Expected: a `path_err … mean` / `joint_err … mean` pair to compare against 109.8 / 66.8.

- [ ] **Step 4: Sweep the averaging window**

Repeat Steps 2–3 for {60,70}, {40,50,60,70}, {30,40,50,60,70}. Record each mean. Keep the best window.

- [ ] **Step 5: Commit results**

```bash
scp 'hikvm:~/HUMOF/results/hik-releaseV0.1/err_avg_*.csv' /opt/study/ail/hik/docs/results/humof-improve/
cd /opt/study/ail/hik && git add docs/results/humof-improve/ tools 2>/dev/null; git add docs/results/humof-improve/ && git commit -m "feat(humof): Phase 1a checkpoint-averaging eval results"
```

---

## Task 3: Phase 1b — test-time rotation averaging (TTA)

**Files:**
- Modify (VM): `~/HUMOF/main.py` eval branch (the `else:` block ~lines 309-320, plus the model call ~267-282)

- [ ] **Step 1: Add a TTA rotation helper (z-axis only, no translation)**

Mirror `datasets/aug.py::A`'s z-rotation but with FIXED angles and inverse-rotate the output. Add to `main.py` (near the top, after imports):

```python
import math
def _zrot(theta, dtype, device):
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=dtype, device=device)
    return R
# rotate points p (..,3) by R:  p @ R.T
```

- [ ] **Step 2: Wrap the eval prediction in a rotation-average (guarded by a `TTA_ANGLES` flag)**

In the eval-only branch, run the model once per angle on the rotated inputs and average the inverse-rotated predictions. The model consumes `joints_repeat_O`, `dcts`, `scene`, `primary`, `others` — all in world coords. For TTA, rotate the raw joints/scene BEFORE `pipelines.Preprocess`, or (simpler, less invasive) rotate only the final prediction frame-of-reference is NOT valid (DCT is precomputed). **Correct hook:** rotate `pose`/`objs_pc`/`others` at the top of the loop iteration (before `Preprocess.A_multiPerson`), then inverse-rotate `y` before error accumulation. Pseudocode inserted at loop top and around the model call:

```python
TTA_ANGLES = [0.0]  # baseline; set e.g. [0, math.pi/2, math.pi, 3*math.pi/2] to enable TTA
...
# accumulate averaged prediction across angles
y_acc = None
for theta in TTA_ANGLES:
    R = _zrot(theta, pose.dtype, pose.device)
    pose_r   = pose @ R.T
    others_r = (others_world @ R.T) if others_world is not None else None
    objs_r   = objs_pc.clone(); objs_r[..., :3] = objs_pc[..., :3] @ R.T
    # ...build inputs from (pose_r, objs_r, others_r) via the SAME Preprocess path...
    # run model -> y_theta (B,T,J,3) in rotated frame
    y_theta_world = y_theta @ R   # inverse rotation (R.T.T == R)
    y_acc = y_theta_world if y_acc is None else y_acc + y_theta_world
y = y_acc / len(TTA_ANGLES)
```
**Note for implementer:** the cleanest, least-error-prone realization is a small refactor that extracts "given world-frame pose/scene/others → predicted world-frame y" into one function, then calls it per angle. Verify correctness with the single-angle identity check in Step 3 BEFORE trusting multi-angle numbers.

- [ ] **Step 3: Identity check — TTA with `[0.0]` must equal the plain baseline**

```bash
ssh hikvm 'cd ~/HUMOF && conda run -n humof env TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=$CONDA_PREFIX python main.py 2>&1 | tail -6'  # with TTA_ANGLES=[0.0], cp_iter=70
```
Expected: `path_err/joint_err mean` == 109.8 / 66.8 exactly (float tolerance). If not, the rotate/inverse-rotate wiring is wrong — fix before proceeding. This is the critical correctness gate.

- [ ] **Step 4: Enable 4-angle TTA and eval**

Set `TTA_ANGLES=[0, pi/2, pi, 3pi/2]`, cp_iter=70, run. Save `err_tta4_ckpt70.csv`. Compare mean vs. baseline.

- [ ] **Step 5: Combine best checkpoint-average + TTA**

Point eval at the best averaged checkpoint from Task 2 with 4-angle TTA enabled. Save `err_avgbest_tta4.csv`. This is the combined Phase-1 number.

- [ ] **Step 6: Commit results**

```bash
scp 'hikvm:~/HUMOF/results/hik-releaseV0.1/err_tta*.csv' 'hikvm:~/HUMOF/results/hik-releaseV0.1/err_avgbest_tta4.csv' /opt/study/ail/hik/docs/results/humof-improve/
cd /opt/study/ail/hik && git add docs/results/humof-improve/ && git commit -m "feat(humof): Phase 1b TTA rotation-averaging + combined results"
```

---

## Task 4: Phase 1 write-up + stop VM

**Files:** Create `hik/docs/results/humof-improve/phase1-results.md`

- [ ] **Step 1: Tabulate all Phase-1 numbers**

Table: baseline (109.8/66.8) vs. each checkpoint-average window vs. TTA-4 vs. combined, columns path/pose per horizon (0.5/1/1.5/2s) + mean. Note which gains exceed a ~1–2mm honesty threshold.

- [ ] **Step 2: Stop the VM (cost)**

```bash
gcloud compute instances stop hik-simlpe-train --zone=asia-southeast1-b
```

- [ ] **Step 3: Commit + update memory**

```bash
cd /opt/study/ail/hik && git add docs/results/humof-improve/phase1-results.md && git commit -m "docs(humof): Phase 1 results write-up"
```
Update memory `humof-repro-plan.md` with the Phase-1 outcome and whether it cleared the honesty threshold.

---

## Task 5: Phase 2 decision gate (STOP — do not auto-proceed)

**Not a code task.** After Phases 0–1, review with the user before spending a ~24h retrain:
- If Phase 1 already gives a clean, defensible improvement + the subset finding → the course deliverable may be complete; Phase 2 is optional upside.
- If proceeding: pick the lever from evidence — **iterative multi-person refinement** (paper future-work #3) by default, or **velocity/accel input features** if Phase 0/failure analysis shows abrupt-motion cases dominate the residual. Then return to brainstorming/writing-plans for a Phase-2-specific plan (it's a real architecture change with its own TDD-able pieces).

---

## Self-Review notes
- Spec Phase 0 → Task 1; Phase 1a → Task 2; Phase 1b → Task 3; Phase 1 write-up → Task 4; Phase 2 gate → Task 5. All spec sections covered.
- The single biggest correctness risk is the TTA rotate/inverse-rotate wiring — Task 3 Step 3 is an explicit identity gate before any multi-angle claim.
- Exact conf line numbers are intentionally discovered in Task 0 (local clone is not authoritative), not hard-coded — this is a deliberate choice, not a placeholder.
