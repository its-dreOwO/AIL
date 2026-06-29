import torch

from forecasting.humanmac import HumanMACDenoiser, sinusoidal_embedding


def test_sinusoidal_embedding_shape():
    t = torch.arange(5)
    emb = sinusoidal_embedding(t, 16)
    assert emb.shape == (5, 16)
    assert torch.isfinite(emb).all()


def test_denoiser_forward_shape():
    model = HumanMACDenoiser(pose_dim=87, n_coeff=20, d_model=32, n_layers=2, n_heads=4)
    X = torch.randn(3, 20, 87)
    t = torch.randint(0, 1000, (3,))
    out = model(X, t)
    assert out.shape == (3, 20, 87)
    assert torch.isfinite(out).all()


def test_denoiser_output_depends_on_timestep():
    torch.manual_seed(0)
    model = HumanMACDenoiser(pose_dim=87, n_coeff=20, d_model=32, n_layers=2, n_heads=4)
    X = torch.randn(2, 20, 87)
    out_a = model(X, torch.zeros(2, dtype=torch.long))
    out_b = model(X, torch.full((2,), 999, dtype=torch.long))
    assert not torch.allclose(out_a, out_b)


from forecasting.humanmac import HumanMAC


def _tiny_humanmac():
    return HumanMAC(
        pose_dim=87, window=8, n_in=4, n_coeff=8,
        d_model=16, n_layers=1, n_heads=2, timesteps=20, ddim_steps=4,
    )


def test_humanmac_forward_is_finite_scalar_loss():
    model = _tiny_humanmac()
    x0 = torch.randn(2, 8, 87)
    loss = model(x0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_humanmac_sample_shape_and_stochastic():
    torch.manual_seed(0)
    model = _tiny_humanmac()
    obs = torch.randn(4, 87)  # n_in=4
    a = model.sample(obs, k=3)
    b = model.sample(obs, k=3)
    assert a.shape == (3, 4, 87)  # k, n_out=4, pose_dim
    assert torch.isfinite(a).all()
    assert not torch.allclose(a, b)
