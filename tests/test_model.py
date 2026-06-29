import torch

from forecasting.model import SceneConditionedSiMLPe, SiMLPe
from forecasting.scene_features import SCENE_FEATURE_DIM


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


def test_velocity_mode_integrates_framewise_deltas_from_last_frame():
    model = SiMLPe(t_in=250, t_out=250, pose_dim=87, n_blocks=1, output_mode="velocity")
    torch.nn.init.zeros_(model.motion_fc_out.weight)
    torch.nn.init.constant_(model.motion_fc_out.bias, 0.25)
    x = torch.randn(2, 250, 87)

    y = model(x)

    steps = torch.arange(1, 251, dtype=x.dtype).reshape(1, 250, 1)
    expected = x[:, -1:, :] + steps * 0.25
    assert torch.allclose(y, expected, atol=1e-5)


def test_rejects_unknown_output_mode():
    try:
        SiMLPe(t_in=250, t_out=250, pose_dim=87, output_mode="bogus")
    except ValueError as exc:
        assert "output_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scene_conditioned_forward_shape():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=2, scene_dim=SCENE_FEATURE_DIM
    )
    x = torch.randn(4, 250, 87)
    scene = torch.randn(4, SCENE_FEATURE_DIM)

    y = model(x, scene)

    assert y.shape == (4, 250, 87)
    assert torch.isfinite(y).all()


def test_scene_conditioned_output_changes_with_scene_features():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=1, scene_dim=SCENE_FEATURE_DIM
    )
    x = torch.randn(1, 250, 87)
    scene_a = torch.zeros(1, SCENE_FEATURE_DIM)
    scene_b = torch.ones(1, SCENE_FEATURE_DIM)

    y_a = model(x, scene_a)
    y_b = model(x, scene_b)

    assert not torch.allclose(y_a, y_b)


def test_scene_conditioned_position_mode_anchors_on_last_frame_when_head_zero():
    model = SceneConditionedSiMLPe(
        t_in=250, t_out=250, pose_dim=87, n_blocks=1, scene_dim=SCENE_FEATURE_DIM
    )
    torch.nn.init.zeros_(model.motion_fc_out.weight)
    torch.nn.init.zeros_(model.motion_fc_out.bias)
    x = torch.randn(2, 250, 87)
    scene = torch.randn(2, SCENE_FEATURE_DIM)

    y = model(x, scene)

    assert torch.allclose(y, x[:, -1:, :].expand(-1, 250, -1), atol=1e-5)
