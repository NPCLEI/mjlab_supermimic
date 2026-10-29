"""Learned reward terms: for reward terms that need simulator-only quantities.

``foot_clearance`` reads foot geom heights, ``foot_slip`` reads contact point velocities, ``air_time``
reads contact history -- none exist inside an imagined rollout. Rather than hand-writing a proxy
(which measurably optimises the wrong objective: correlation 0.56 with the real reward), train a head
``r̂_k(s, a, xi)`` supervised by the per-term values the environment already computes.

Three rules learned the hard way:

* the head is **pre-trained before the policy may use it** (using an untrained head to score imagined
  rollouts collapses the policy);
* its output is **clamped** to ``mean ± kσ`` of the observed values, so one bad prediction cannot
  dominate a rollout;
* targets are **standardised** with running statistics so the regression is well conditioned.

Like the analytic projections, every value this module returns is already ``weight * dt``-scaled, i.e.
the same number the environment reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LearnedRewardCfg:
  hidden: tuple[int, ...] = (256, 256)
  lr: float = 3e-4
  batch_size: int = 1024
  updates_per_iteration: int = 1
  #: no policy update may consume the head until its correlation reaches this
  min_corr: float = 0.9
  clip_sigma: float = 5.0
  warmup_iterations: int = 200
  weight_decay: float = 0.0


class LearnedRewardGroup(nn.Module):
  """One small MLP per learned term (sharing a trunk would couple unrelated terms)."""

  def __init__(
    self,
    names: tuple[str, ...],
    weights: tuple[float, ...],
    dt: float,
    state_dim: int = 71,
    act_dim: int = 29,
    xi_dim: int = 33,
    hidden: tuple[int, ...] = (256, 256),
  ) -> None:
    super().__init__()
    self.names = names
    self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
    self.dt = float(dt)
    self.heads = nn.ModuleList()
    for _ in names:
      dims = [state_dim + act_dim + xi_dim, *hidden, 1]
      layers: list[nn.Module] = []
      for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
      self.heads.append(nn.Sequential(*layers[:-1]))
    self.register_buffer("in_mean", torch.zeros(state_dim + act_dim + xi_dim))
    self.register_buffer("in_var", torch.ones(state_dim + act_dim + xi_dim))
    self.register_buffer("out_mean", torch.zeros(len(names)))
    self.register_buffer("out_std", torch.ones(len(names)))
    self.register_buffer("corr", torch.zeros(len(names)))
    self._fitted = False

  # ------------------------------------------------------------------ inference
  def _features(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor | None) -> torch.Tensor:
    if xi is None:
      xi = torch.zeros(*s.shape[:-1], self.in_mean.numel() - s.shape[-1] - a.shape[-1], device=s.device, dtype=s.dtype)
    x = torch.cat([s, a, xi], dim=-1)
    return (x - self.in_mean) / self.in_var.clamp_min(1e-5).sqrt()

  def forward(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    x = self._features(s, a, xi)
    out: dict[str, torch.Tensor] = {}
    total = None
    for k, name in enumerate(self.names):
      raw = self.heads[k](x).squeeze(-1)
      value = raw * self.out_std[k] + self.out_mean[k]
      value = value.clamp(
        self.out_mean[k] - 5.0 * self.out_std[k], self.out_mean[k] + 5.0 * self.out_std[k]
      )
      scaled = self.dt * self.weights[k] * value
      out[name] = scaled
      total = scaled if total is None else total + scaled
    out["total"] = total
    return out

  # ------------------------------------------------------------------ fitting
  @torch.no_grad()
  def _update_stats(self, x: torch.Tensor, y: torch.Tensor, momentum: float = 0.05) -> None:
    xm, xv = x.mean(0), x.var(0, unbiased=False)
    ym, ys = y.mean(0), y.std(0).clamp_min(1e-6)
    if not self._fitted:
      self.in_mean.copy_(xm)
      self.in_var.copy_(xv)
      self.out_mean.copy_(ym)
      self.out_std.copy_(ys)
      self._fitted = True
    else:
      self.in_mean.mul_(1 - momentum).add_(xm, alpha=momentum)
      self.in_var.mul_(1 - momentum).add_(xv, alpha=momentum)
      self.out_mean.mul_(1 - momentum).add_(ym, alpha=momentum)
      self.out_std.mul_(1 - momentum).add_(ys, alpha=momentum)

  def update(self, buffer, cfg: LearnedRewardCfg, optim: torch.optim.Optimizer) -> dict[str, float]:
    """Fit to the per-term values logged by the environment."""
    w = buffer.sample_windows(min(cfg.batch_size, int(buffer.ptr.min())), 1)
    if getattr(w, "r_terms", None) is None:
      return {}
    s, a, xi, target = w.states[0], w.a_exec[0], w.xi, w.r_terms[0]
    x = torch.cat([s, a, xi], dim=-1)
    self._update_stats(x, target)
    xn = (x - self.in_mean) / self.in_var.clamp_min(1e-5).sqrt()
    yn = (target - self.out_mean) / self.out_std
    loss = None
    for k in range(len(self.names)):
      l = F.mse_loss(self.heads[k](xn).squeeze(-1), yn[:, k])
      loss = l if loss is None else loss + l
    loss = loss / len(self.names)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
    optim.step()
    with torch.no_grad():
      preds = torch.stack([self.heads[k](xn).squeeze(-1) for k in range(len(self.names))], dim=-1)
      preds = preds * self.out_std + self.out_mean
      for k in range(len(self.names)):
        t = target[:, k]
        if float(t.std()) > 1e-8:
          self.corr[k] = torch.corrcoef(torch.stack([preds[:, k], t]))[0, 1]
    return {
      "learned_mse": float(loss),
      "learned_corr_min": float(self.corr.min()),
      **{f"corr/{n}": float(self.corr[k]) for k, n in enumerate(self.names)},
    }

  def ready(self, min_corr: float) -> bool:
    return self._fitted and bool((self.corr >= min_corr).all())
