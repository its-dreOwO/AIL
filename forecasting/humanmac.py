import math

import torch
import torch.nn as nn


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
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, pose_dim)

    def forward(self, X, t):
        temb = self.t_mlp(sinusoidal_embedding(t, self.d_model)).unsqueeze(1)
        h = self.in_proj(X) + self.pos + temb
        h = self.encoder(h)
        return self.out(h)
