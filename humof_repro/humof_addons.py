"""Reconstructed `primary_filterC` and `filterB` for HUMOF's HIK pipeline.

These two functions are referenced by datasets/dataset_hik.py (lines 148, 233)
but were NEVER COMMITTED to the HUMOF repo — they are absent from its entire git
history and from the SAST repo it depends on. They have to be reconstructed from
their call sites, their thresholds (datasets/dataset_preprocess/hik/confs.py),
and their observable effect on the dataset size.

This is the project's headline reproducibility limitation: `primary_filterC`
alone decides ~62% of the eval subset's composition on recording D, so absolute
numbers are NOT comparable to the paper's Table 1.

VERIFICATION TARGETS (recording D, mode='test', step=41, t_total=75), taken from
the Phase 0 characterization (docs/results/humof-improve/subset-report.md):

    __ct_filtered_1 (not present at all)   = 69,290
    __ct_filtered_2 (partial presence)     =    284
    fully-present candidates               = 34,370
      - rejected by primary_filterC        = 21,186   (61.6% of candidates)
      - rejected by filterB / no others    =     22
    admitted (len(dataset))                = 13,162

Run humof_repro/verify_filters.py to check these. If they do not reproduce, the
comparison against our recorded baseline (path 109.71 / pose 66.81) is invalid.

Semantics (reconstructed):
  primary_filterC(primary, thres) -> True means REJECT this target window.
      Rejects near-static targets: takes each joint's axis-aligned bounding box
      over the window, measures its diagonal, and keeps the window only if the
      most-moving joint's bbox diagonal reaches `thres` metres.
      thres = 0.4 (test) / 0.6 (train).

  filterB(other, primary, thres, use_cuda) -> True means REJECT this other person.
      Keeps an "other" only if their root joint comes within `thres` metres of
      the target's root at some frame in the window.
      thres = 8.0 (test) / 9.9 (train). These are large relative to a kitchen,
      which is why filterB is nearly a no-op on the target set.

Both must accept either numpy arrays or CUDA torch tensors: dataset_hik.py moves
the joint arrays to the GPU when ACCE_filterB_by_cuda is set.
"""

import numpy as np
import torch

ROOT_JOINT_IDX = 0


def _bbox_diag_per_joint(x):
    """x: (T, J, 3) -> (J,) axis-aligned bbox diagonal of each joint's track."""
    if isinstance(x, torch.Tensor):
        extent = x.amax(dim=0) - x.amin(dim=0)          # (J, 3)
        return torch.linalg.norm(extent, dim=-1)        # (J,)
    extent = x.max(axis=0) - x.min(axis=0)              # (J, 3)
    return np.linalg.norm(extent, axis=-1)              # (J,)


def primary_filterC(primary, thres):
    """True => filter out (reject) this target window as too static.

    primary: (t_total, J, 3) np.ndarray or torch.Tensor
    """
    diag = _bbox_diag_per_joint(primary)
    if isinstance(diag, torch.Tensor):
        max_diag = diag.max().item()
    else:
        max_diag = float(diag.max())
    return max_diag < float(thres)


def filterB(other, primary, thres, use_cuda=False):
    """True => filter out (reject) this "other" person as too far away.

    other, primary: (t_total, J, 3) np.ndarray or torch.Tensor
    """
    root_other = other[:, ROOT_JOINT_IDX]      # (T, 3)
    root_primary = primary[:, ROOT_JOINT_IDX]  # (T, 3)
    if isinstance(root_other, torch.Tensor) or isinstance(root_primary, torch.Tensor):
        if not isinstance(root_other, torch.Tensor):
            root_other = torch.as_tensor(root_other, device=root_primary.device)
        if not isinstance(root_primary, torch.Tensor):
            root_primary = torch.as_tensor(root_primary, device=root_other.device)
        min_dist = torch.linalg.norm(root_other - root_primary, dim=-1).min().item()
    else:
        min_dist = float(np.linalg.norm(root_other - root_primary, axis=-1).min())
    return min_dist > float(thres)
