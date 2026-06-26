import torch

from forecasting.model import SiMLPe


def test_forward_shape():
    model = SiMLPe(t_in=250, t_out=250, pose_dim=87, n_blocks=4)
    x = torch.randn(4, 250, 87)
    y = model(x)
    assert y.shape == (4, 250, 87)
    assert torch.isfinite(y).all()


def test_residual_anchors_on_last_frame():
    model = SiMLPe(t_in=250, t_out=250, pose_dim=87, n_blocks=2)
    torch.nn.init.zeros_(model.motion_fc_out.weight)
    torch.nn.init.zeros_(model.motion_fc_out.bias)
    x = torch.randn(2, 250, 87)
    y = model(x)
    last = x[:, -1:, :].expand(-1, 250, -1)
    assert torch.allclose(y, last, atol=1e-5)
