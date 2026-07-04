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


@torch.no_grad()
def ddim_sample(
    denoiser,
    diffusion,
    shape,
    obs_full_time,
    mask_time,
    dct_fn,
    idct_fn,
    ddim_steps,
    device="cpu",
    x0_clip=None,
):
    B = shape[0]
    abar = diffusion.alphas_cumprod.to(device)
    obs_full_time = obs_full_time.to(device)
    mask_time = mask_time.to(device)
    obs_latent = dct_fn(obs_full_time)

    X = torch.randn(shape, device=device)
    seq = torch.linspace(diffusion.timesteps - 1, 0, ddim_steps).round().long().tolist()
    seq_prev = seq[1:] + [-1]

    for t, t_prev in zip(seq, seq_prev):
        tb = torch.full((B,), t, device=device, dtype=torch.long)
        eps = denoiser(X, tb)
        a_t = abar[t]
        x0 = (X - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        # Static thresholding: at the first steps abar ~ 0, so the x0 estimate
        # divides by sqrt(abar) ~ 5e-5 and explodes ~20000x. Clip it back to the
        # data range so the trajectory stays on-manifold (cf. DDPM/Imagen).
        if x0_clip is not None:
            x0 = x0.clamp(-x0_clip, x0_clip)
        a_prev = abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        X = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps

        # masked completion: pin observed region at noise level t_prev (clean at end)
        level = max(t_prev, 0)
        known_latent = (
            abar[level].sqrt() * obs_latent
            + (1.0 - abar[level]).sqrt() * torch.randn_like(obs_latent)
        )
        x_time = idct_fn(X)
        known_time = idct_fn(known_latent)
        x_time = mask_time * known_time + (1.0 - mask_time) * x_time
        X = dct_fn(x_time)

    out = idct_fn(X)
    out = mask_time * obs_full_time + (1.0 - mask_time) * out  # hard pin observed
    return out
