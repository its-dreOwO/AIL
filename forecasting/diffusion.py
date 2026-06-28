import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).to(torch.float32)


def _gather(values, t, ndim):
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, t, noise):
        abar = _gather(self.alphas_cumprod, t, x0.ndim)
        return abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise

    def p_losses(self, denoiser, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = denoiser(xt, t)
        return F.mse_loss(pred, noise)
