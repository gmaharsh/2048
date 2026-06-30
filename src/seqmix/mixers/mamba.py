"""Pure-PyTorch reference Mamba (selective state-space model).

Runs anywhere (CPU/GPU) without the CUDA-only ``mamba-ssm`` kernels. The
selective scan is implemented as an explicit sequential recurrence -- correct
and differentiable, just slower than the fused kernel. For large GPU runs you
can swap in the official kernel; this reference is what makes the controlled
study reproducible on a single machine.

The short causal convolution is exposed as ``d_conv`` / ``use_conv`` so the
mechanistic ablation in Phase 4 (its role in associative recall) is a one-line
config change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .base import SequenceMixer


class MambaMixer(SequenceMixer):
    is_attention = False

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        mk = config.mixer_kwargs
        self.d_model = config.d_model
        self.expand = float(mk.get("expand", 2.0))
        self.d_inner = int(self.expand * self.d_model)
        self.d_state = int(mk.get("d_state", 16))
        self.d_conv = int(mk.get("d_conv", 4))
        self.use_conv = bool(mk.get("use_conv", True)) and self.d_conv > 1
        self.dt_rank = int(mk.get("dt_rank", max(1, math.ceil(self.d_model / 16))))

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        if self.use_conv:
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner, kernel_size=self.d_conv,
                groups=self.d_inner, padding=self.d_conv - 1, bias=True,
            )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))          # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    # ---- helpers -----------------------------------------------------------
    def _ssm_params(self, x: torch.Tensor):
        """From conv'd x (B,T,d_inner) compute (delta, A, B, C)."""
        x_dbl = self.x_proj(x)  # (B,T,dt_rank+2*d_state)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))          # (B,T,d_inner)
        A = -torch.exp(self.A_log.float())               # (d_inner,d_state)
        return delta, A, B, C

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return x
        xt = x.transpose(1, 2)                           # (B,d_inner,T)
        xt = self.conv1d(xt)[..., : x.shape[1]]
        return xt.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xb, z = xz.chunk(2, dim=-1)                       # each (B,T,d_inner)
        xb = F.silu(self._conv(xb))

        delta, A, Bmat, Cmat = self._ssm_params(xb)
        # Discretise and scan: h_t = exp(delta*A) h_{t-1} + delta*B*x_t
        deltaA = torch.exp(delta.unsqueeze(-1) * A)       # (B,T,d_inner,d_state)
        deltaBx = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * xb.unsqueeze(-1)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            h = deltaA[:, t] * h + deltaBx[:, t]
            ys.append((h * Cmat[:, t].unsqueeze(1)).sum(-1))  # (B,d_inner)
        y = torch.stack(ys, dim=1)                        # (B,T,d_inner)
        y = y + xb * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    def analytic_state_bytes(self, seq_len: int, dtype_bytes: int = 2) -> int:
        # Recurrent state is independent of sequence length.
        ssm = self.d_inner * self.d_state
        conv = self.d_inner * (self.d_conv - 1) if self.use_conv else 0
        return (ssm + conv) * dtype_bytes

    def init_cache(self, batch_size: int, device, dtype=torch.float32):
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return {"conv": conv_state, "ssm": ssm_state}

    def step(self, x_t: torch.Tensor, cache, pos: int):
        B = x_t.shape[0]
        xz = self.in_proj(x_t)                            # (B,1,2*d_inner)
        xb, z = xz.chunk(2, dim=-1)
        xb = xb.squeeze(1)                                # (B,d_inner)

        if self.use_conv:
            conv_state = cache["conv"]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = xb
            cache["conv"] = conv_state
            w = self.conv1d.weight.squeeze(1)             # (d_inner,d_conv)
            xb = (conv_state * w).sum(-1) + self.conv1d.bias
        xb = F.silu(xb)

        x_dbl = self.x_proj(xb)
        delta, Bmat, Cmat = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))           # (B,d_inner)
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(delta.unsqueeze(-1) * A)       # (B,d_inner,d_state)
        deltaBx = delta.unsqueeze(-1) * Bmat.unsqueeze(1) * xb.unsqueeze(-1)
        ssm_state = deltaA * cache["ssm"] + deltaBx
        cache["ssm"] = ssm_state
        y = (ssm_state * Cmat.unsqueeze(1)).sum(-1) + xb * self.D
        y = y * F.silu(z.squeeze(1))
        return self.out_proj(y).unsqueeze(1), cache
