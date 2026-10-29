"""Running input normalisation (no parameters, updated from real transitions only)."""

from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Standardise the last dimension with exponential running statistics."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.01) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps
        self.momentum = momentum

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # a single non-finite transition (a fallen robot) must not poison the statistics
        flat = torch.nan_to_num(x.detach().reshape(-1, x.shape[-1]), nan=0.0, posinf=0.0, neginf=0.0)
        mean, var = flat.mean(0), flat.var(0, unbiased=False)
        if float(self.count) == 0.0:
            self.mean.copy_(mean)
            self.var.copy_(var)
            self.count.fill_(1.0)
        else:
            m = self.momentum
            self.mean.mul_(1.0 - m).add_(mean, alpha=m)
            self.var.mul_(1.0 - m).add_(var, alpha=m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x - self.mean) / self.var.clamp_min(self.eps).sqrt()
