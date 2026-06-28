import torch

from forecasting.diffusion import GaussianDiffusion, cosine_beta_schedule


def test_cosine_schedule_betas_in_range_and_alphacumprod_decreasing():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas > 0) and torch.all(betas <= 0.999)
    abar = torch.cumprod(1.0 - betas, dim=0)
    assert torch.all(abar[1:] <= abar[:-1] + 1e-6)
    assert abar[0] <= 1.0 and abar[-1] > 0.0


def test_q_sample_is_clean_at_t0_and_noisy_at_tmax():
    diff = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(4, 10, 87)
    noise = torch.randn_like(x0)
    t0 = torch.zeros(4, dtype=torch.long)
    xt0 = diff.q_sample(x0, t0, noise)
    assert (xt0 - x0).abs().mean() < 1e-2

    tmax = torch.full((4,), 999, dtype=torch.long)
    xtmax = diff.q_sample(x0, tmax, noise)
    assert (xtmax - x0).abs().mean() > (xt0 - x0).abs().mean()


def test_ddim_sample_pins_observed_region_with_full_dct():
    from forecasting.diffusion import ddim_sample

    # Full (non-truncated) DCT so dct->idct is identity and masking is exact.
    T, C, L = 8, 3, 8
    dct_m = torch.linalg.qr(torch.randn(T, T))[0]  # orthonormal TxT
    idct_m = dct_m.t()

    def dct_fn(x):  # [B,T,C] -> [B,L,C]
        return torch.einsum("lt,btc->blc", dct_m[:L], x)

    def idct_fn(X):  # [B,L,C] -> [B,T,C]
        return torch.einsum("tl,blc->btc", idct_m[:, :L], X)

    diff = GaussianDiffusion(timesteps=20)
    denoiser = lambda xt, t: torch.zeros_like(xt)

    n_in = 4
    obs = torch.randn(2, T, C)
    obs[:, n_in:] = 0.0
    mask = torch.zeros(T, 1)
    mask[:n_in] = 1.0

    out = ddim_sample(
        denoiser, diff, (2, L, C), obs, mask, dct_fn, idct_fn, ddim_steps=5
    )
    assert out.shape == (2, T, C)
    assert torch.allclose(out[:, :n_in], obs[:, :n_in], atol=1e-4)
    assert torch.isfinite(out).all()


def test_p_losses_returns_finite_scalar_with_grad():
    diff = GaussianDiffusion(timesteps=50)

    class Echo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(87, 87)

        def forward(self, xt, t):
            return self.w(xt)

    denoiser = Echo()
    x0 = torch.randn(3, 10, 87)
    t = torch.randint(0, 50, (3,))
    loss = diff.p_losses(denoiser, x0, t)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert denoiser.w.weight.grad is not None
