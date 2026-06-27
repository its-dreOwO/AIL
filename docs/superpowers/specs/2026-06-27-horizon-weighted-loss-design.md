# Horizon-Weighted Loss + Tunable Training Knobs — Design

**Date:** 2026-06-27
**Status:** Approved (brainstorm), pending implementation plan
**Scope:** loss + hyperparameters only — no model, eval, or data changes

## Motivation

The first full siMLPe baseline (trained A+B+C+D, stepsize 50, 80 epochs, Tesla T4)
**loses to the zero-velocity baseline at every horizon**, including @1s where a
competent short-term model should win:

| horizon | zero-velocity | siMLPe |
|---------|--------------:|-------:|
| overall | **1.108**     | 1.188  |
| @1s     | **0.520**     | 0.607  |
| @5s     | **1.254**     | 1.351  |
| @10s    | **1.422**     | 1.472  |

Validation loss flatlined early (0.106 @ epoch 29 → 0.104 @ epoch 80). Diagnosis:
**underfitting / mean-collapse**, not a denormalization bug. The model is currently
trained with a flat, unweighted MPJPE averaged over all 250 future frames. At long
horizons the L2-optimal prediction genuinely *is* the mean pose, so the model is
correctly minimizing its objective by collapsing toward "last frame + small noise."
More epochs at the same objective will not escape that minimum.

This iteration changes the **training objective and hyperparameters** to reward
near-term accuracy and reduce the mean-collapse incentive. The model architecture,
evaluation metric, and data pipeline are deliberately left untouched so the
scoreboard comparison against `1.108` stays fair.

## Non-goals

- No changes to `model.py` (no output reframing / velocity-delta representation).
- No changes to `callbacks.py`, `data.py`, `evaluate.py`, or `metrics.py`.
- No changes to `config.py` structural constants (`N_IN`, `N_OUT`, `POSE_DIM`, …).
- Not a hyperparameter *sweep harness* — just expose the knobs and set sensible
  new defaults.

## Design

### 1. Weighted loss (`forecasting/losses.py`)

Add a pure helper and make both loss functions accept optional per-frame weights.
`weights=None` reproduces today's exact behavior (a plain `.mean()`), so existing
behavior remains the default and is regression-testable.

```python
def horizon_weights(t, floor, device=None):   # returns [t] tensor
    # w_k = 1 - (1 - floor) * k / (t - 1)
    # floor == 1.0  -> all ones (uniform, == status quo)
    # floor  < 1.0  -> linear decay from 1.0 at k=0 down to floor at k=t-1

def mpjpe_loss(pred, target, weights=None):    # weights: [T]   or None
def velocity_loss(pred, target, weights=None): # weights: [T-1] or None
```

**Mechanics.** `_per_joint_l2(pred, target)` already returns a `[b, t, J]` curve.
- `weights is None` → mean over all axes (unchanged).
- `weights` given → mean over `b, J` to a `[t]` per-frame curve `c`, then a
  **weighted average** over time: `Σ_k w_k · c_k / Σ_k w_k`.

**Velocity term.** The velocity curve is over `T-1` diff frames. It uses its own
`horizon_weights(T-1, floor)` so the *same* `floor` applies to the shorter curve.
Decision: apply the **same** horizon weighting to both the position and velocity
terms — both are per-frame curves, and sharing one `floor` keeps a single knob.

### 2. Training knobs (`forecasting/train.py`)

Fix two gaps and add the new lever:

- `vel_weight` is already a `train()` kwarg but `main()` never exposes it — it is
  hard-stuck at `1.0`. Add `--vel-weight`.
- Add `--horizon-floor`.
- `train()` gains `horizon_floor=0.2`. It computes the two weight vectors **once**
  (position length `N_OUT`, velocity length `N_OUT-1`) before the epoch loop and
  threads them into `train_one_epoch` and `eval_loss`, which pass them to the
  losses. Train and val loss use the **same** weighting, so the watched val curve
  reflects the actual objective.

**New recommended defaults** (all overridable on the CLI):

| knob             | old        | new default | rationale |
|------------------|------------|-------------|-----------|
| `lr`             | 3e-3       | **5e-4**    | 3e-3 is high for Adam; consistent with snapping into the flat mean-collapse basin |
| `vel_weight`     | 1.0        | **0.2**     | an equal-weight velocity term fights the position term |
| `horizon_floor`  | — (uniform)| **0.2**     | the new near-term-emphasis lever |

`config.py` is untouched: these are training hyperparameters (like `lr`), not
structural constants.

### 3. Tests (TDD, `forecasting/`)

Pure functions, so straightforward:

- `horizon_weights`: correct length; endpoints `w[0] == 1.0` and `w[-1] == floor`;
  `floor == 1.0` ⇒ all ones.
- **Regression guard:** weighted `mpjpe_loss` with `floor=1.0` equals unweighted
  (within float tolerance). Same for `velocity_loss`.
- **Directional check:** a prediction whose error sits only in *late* frames yields
  a *lower* weighted loss than its unweighted loss; error only in *early* frames
  yields a *higher* weighted loss. One such pair each for `mpjpe_loss` and
  `velocity_loss`.

### 4. Runbook (`forecasting/README.md`)

Update the train command to show `--lr`, `--vel-weight`, `--horizon-floor`, and add
a slot to record the next result against the `1.108` zero-velocity bar.

## Files touched

- `forecasting/losses.py` — `horizon_weights` + weighted loss variants
- `forecasting/train.py` — `horizon_floor` kwarg, weight precompute, new CLI flags, new defaults
- `forecasting/test_losses.py` (new) — unit tests
- `forecasting/README.md` — runbook + result slot

## Success criterion

A retrain with the new defaults clears the zero-velocity baseline
(**overall MPJPE < 1.108**), ideally winning @1s first. If it does not, the next
iteration reframes the model output (deferred, out of scope here).
