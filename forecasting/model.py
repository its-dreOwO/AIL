import torch
import torch.nn as nn

from forecasting.dct import get_dct_matrix


class TemporalMLPBlock(nn.Module):
    """Mix information across time independently for each pose channel."""

    def __init__(self, t: int):
        super().__init__()
        self.fc = nn.Linear(t, t)
        self.ln = nn.LayerNorm(t)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.ln(self.fc(x)))


class SiMLPe(nn.Module):
    """DCT -> temporal MLP blocks -> IDCT -> decoded future positions."""

    def __init__(self, t_in=250, t_out=250, pose_dim=87, n_blocks=4, output_mode="position"):
        super().__init__()
        if t_in != t_out:
            raise AssertionError("this baseline uses square DCT over equal lengths")
        if output_mode not in {"position", "velocity"}:
            raise ValueError("output_mode must be 'position' or 'velocity'")
        self.t_in = t_in
        self.t_out = t_out
        self.pose_dim = pose_dim
        self.output_mode = output_mode

        dct_m, idct_m = get_dct_matrix(t_in)
        self.register_buffer("dct", torch.tensor(dct_m, dtype=torch.float32))
        self.register_buffer("idct", torch.tensor(idct_m, dtype=torch.float32))

        self.motion_fc_in = nn.Linear(pose_dim, pose_dim)
        self.blocks = nn.ModuleList([TemporalMLPBlock(t_in) for _ in range(n_blocks)])
        self.motion_fc_out = nn.Linear(pose_dim, pose_dim)

    def forward(self, x):
        last = x[:, -1:, :]
        h = self.motion_fc_in(x)
        h = torch.einsum("tk,bkc->btc", self.dct, h)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        h = torch.einsum("tk,bkc->btc", self.idct, h)
        h = self.motion_fc_out(h)
        if self.output_mode == "velocity":
            return last + torch.cumsum(h, dim=1)
        return h + last


class SceneConditionedSiMLPe(nn.Module):
    """siMLPe temporal backbone with a per-window scene embedding."""

    def __init__(
        self,
        t_in=250,
        t_out=250,
        pose_dim=87,
        n_blocks=4,
        scene_dim=17,
        scene_hidden=128,
        output_mode="position",
    ):
        super().__init__()
        if t_in != t_out:
            raise AssertionError("this baseline uses square DCT over equal lengths")
        if output_mode not in {"position", "velocity"}:
            raise ValueError("output_mode must be 'position' or 'velocity'")
        self.t_in = t_in
        self.t_out = t_out
        self.pose_dim = pose_dim
        self.output_mode = output_mode

        dct_m, idct_m = get_dct_matrix(t_in)
        self.register_buffer("dct", torch.tensor(dct_m, dtype=torch.float32))
        self.register_buffer("idct", torch.tensor(idct_m, dtype=torch.float32))

        self.motion_fc_in = nn.Linear(pose_dim, pose_dim)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, scene_hidden),
            nn.GELU(),
            nn.Linear(scene_hidden, pose_dim),
        )
        self.blocks = nn.ModuleList([TemporalMLPBlock(t_in) for _ in range(n_blocks)])
        self.motion_fc_out = nn.Linear(pose_dim, pose_dim)

    def forward(self, x, scene_features):
        last = x[:, -1:, :]
        scene_h = self.scene_encoder(scene_features).unsqueeze(1)
        h = self.motion_fc_in(x) + scene_h
        h = torch.einsum("tk,bkc->btc", self.dct, h)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        h = torch.einsum("tk,bkc->btc", self.idct, h)
        h = self.motion_fc_out(h)
        if self.output_mode == "velocity":
            return last + torch.cumsum(h, dim=1)
        return h + last
