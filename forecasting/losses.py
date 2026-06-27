import torch

from forecasting.config import N_JOINTS


def horizon_weights(t, floor, device=None):
    """Linear per-frame weights: 1.0 at frame 0 down to `floor` at frame t-1.

    floor == 1.0 reproduces uniform weighting (all ones).
    """
    k = torch.arange(t, dtype=torch.float32, device=device)
    return 1.0 - (1.0 - floor) * (k / (t - 1))


def _per_joint_l2(pred, target):
    b, t, _ = pred.shape
    p = pred.reshape(b, t, N_JOINTS, 3)
    g = target.reshape(b, t, N_JOINTS, 3)
    return torch.linalg.norm(p - g, dim=-1)


def mpjpe_loss(pred, target):
    return _per_joint_l2(pred, target).mean()


def velocity_loss(pred, target):
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dg = target[:, 1:, :] - target[:, :-1, :]
    return _per_joint_l2(dp, dg).mean()
