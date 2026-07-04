import torch

from forecasting.losses import mpjpe_loss, velocity_loss


def test_mpjpe_zero_when_equal():
    x = torch.randn(3, 250, 87)
    assert mpjpe_loss(x, x).item() == 0.0


def test_mpjpe_positive_and_correct():
    pred = torch.zeros(1, 1, 87)
    target = torch.zeros(1, 1, 87)
    target[0, 0, 0] = 3.0
    target[0, 0, 1] = 4.0
    assert abs(mpjpe_loss(pred, target).item() - 5.0 / 29.0) < 1e-6


def test_velocity_zero_for_constant_motion():
    x = torch.randn(2, 250, 87)
    assert velocity_loss(x, x).item() == 0.0
