import math

import torch
import torch.nn as nn

from forecasting.dct import get_dct_matrix
from forecasting.diffusion import GaussianDiffusion, ddim_sample


def sinusoidal_embedding(t, dim):
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=-1)
    return emb


class HumanMACDenoiser(nn.Module):
    def __init__(self, pose_dim=87, n_coeff=125, d_model=512, n_layers=8, n_heads=8, ff_mult=4):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(pose_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_coeff, d_model))
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * ff_mult,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, pose_dim)

    def forward(self, X, t):
        temb = self.t_mlp(sinusoidal_embedding(t, self.d_model)).unsqueeze(1)
        h = self.in_proj(X) + self.pos + temb
        h = self.encoder(h)
        return self.out(h)


class HumanMAC(nn.Module):
    def __init__(
        self, pose_dim=87, window=500, n_in=250, n_coeff=125,
        d_model=512, n_layers=8, n_heads=8, timesteps=1000, ddim_steps=50,
    ):
        super().__init__()
        self.window = window
        self.n_in = n_in
        self.n_out = window - n_in
        self.n_coeff = n_coeff
        self.ddim_steps = ddim_steps

        dct_m, idct_m = get_dct_matrix(window)
        self.register_buffer("dct_l", torch.tensor(dct_m[:n_coeff], dtype=torch.float32))
        self.register_buffer("idct_l", torch.tensor(idct_m[:, :n_coeff], dtype=torch.float32))

        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.denoiser = HumanMACDenoiser(
            pose_dim=pose_dim, n_coeff=n_coeff, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads,
        )

    def dct(self, x_time):
        return torch.einsum("lt,btc->blc", self.dct_l, x_time)

    def idct(self, X):
        return torch.einsum("tl,blc->btc", self.idct_l, X)

    def forward(self, x0_time):
        X0 = self.dct(x0_time)
        t = torch.randint(0, self.diffusion.timesteps, (X0.shape[0],), device=X0.device)
        return self.diffusion.p_losses(self.denoiser, X0, t)

    @torch.no_grad()
    def sample(self, obs_time, k, ddim_steps=None):
        device = self.dct_l.device
        steps = ddim_steps or self.ddim_steps
        pose_dim = obs_time.shape[-1]
        obs_full = torch.zeros(k, self.window, pose_dim, device=device)
        obs_full[:, : self.n_in] = obs_time.to(device)[None]
        mask = torch.zeros(self.window, 1, device=device)
        mask[: self.n_in] = 1.0
        traj = ddim_sample(
            self.denoiser, self.diffusion, (k, self.n_coeff, pose_dim),
            obs_full, mask, self.dct, self.idct, steps, device=device,
        )
        return traj[:, self.n_in :]
