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
    """DCT -> temporal MLP blocks -> IDCT -> last-frame residual."""

    def __init__(self, t_in=250, t_out=250, pose_dim=87, n_blocks=4):
        super().__init__()
        if t_in != t_out:
            raise AssertionError("this baseline uses square DCT over equal lengths")
        self.t_in = t_in
        self.t_out = t_out
        self.pose_dim = pose_dim

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
        return h + last
