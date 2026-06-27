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


def _weighted_time_mean(curve, weights):
    # curve: [B, T, N_JOINTS]; weights: [T] or None
    if weights is None:
        return curve.mean()
    per_frame = curve.mean(dim=(0, 2))            # [T]
    w = weights.to(per_frame.device)
    return (per_frame * w).sum() / w.sum()


def mpjpe_loss(pred, target, weights=None):
    return _weighted_time_mean(_per_joint_l2(pred, target), weights)


def velocity_loss(pred, target):
    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dg = target[:, 1:, :] - target[:, :-1, :]
    return _per_joint_l2(dp, dg).mean()
