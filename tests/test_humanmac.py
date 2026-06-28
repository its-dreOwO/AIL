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
