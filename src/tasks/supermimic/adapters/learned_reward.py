"""Learned reward model: imagination scores states with r̂(s, a, xi) instead of hand-written terms.

Why this exists: a motion-tracking reward is eight analytic kernels on reference bodies, so it can
be re-implemented exactly. A velocity task's reward is 14 terms of which six need contact forces
(``foot_clearance`` -2.0, ``foot_slip``, ``air_time``, ``soft_landing``, ``self_collisions``,
``angular_momentum``). Hand-writing the other eight gives a reward that is only 0.56-correlated
with the environment's, 3.8x too large and blind to penalties as large as -0.22 per step, so the
policy optimises the wrong objective and degrades (observed: return -0.96 -> -1.38).

The model is trained on the replay buffer's *real* rewards, which is exactly the supervision PPO
gets for free. It wraps the analytic heads and delegates every other hook, so the engine is
untouched:

    heads = LearnedRewardHeads(analytic_heads, model, cfg)   # wm.heads = heads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class RewardModelCfg:
  hidden: tuple[int, ...] = (256, 256)
  lr: float = 3e-4
  batch_size: int = 1024
  updates_per_iteration: int = 1
  #: normalise the target by this (the runner sets it to the task's max step reward)
  target_scale: float = 1.0
  #: predict standardised targets (mean/std tracked from the buffer) -- keeps the net well scaled
  normalize: bool = True
  #: clamp the imagined reward to mean +- this many standard deviations
  clip_sigma: float = 5.0
  weight_decay: float = 0.0


class RewardModel(nn.Module):
  """r̂(s_t, a_t, xi_t) -> scalar reward."""

  def __init__(self, state_dim: int = 71, act_dim: int = 29, xi_dim: int = 33, hidden: tuple[int, ...] = (256, 256)) -> None:
    super().__init__()
    self.state_dim, self.act_dim, self.xi_dim = state_dim, act_dim, xi_dim
    dims = [state_dim + act_dim + xi_dim, *hidden, 1]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
      layers += [nn.Linear(a, b), nn.SiLU()]
    layers = layers[:-1]  # no activation on the output
    self.net = nn.Sequential(*layers)
    # features: per-term magnitudes differ by orders of magnitude, so scale them
    self.register_buffer("state_scale", torch.ones(state_dim))
    self.register_buffer("act_scale", torch.ones(act_dim))
    self.register_buffer("xi_scale", torch.ones(xi_dim))

  def forward(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor | None) -> torch.Tensor:
    parts = [s / self.state_scale, a / self.act_scale]
    if xi is not None:
      parts.append(xi / self.xi_scale)
    else:
      parts.append(torch.zeros(*s.shape[:-1], self.xi_dim, device=s.device, dtype=s.dtype))
    return self.net(torch.cat(parts, dim=-1)).squeeze(-1)

  @torch.no_grad()
  def set_scales_from(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor | None) -> None:
    self.state_scale.copy_(s.abs().mean(0).clamp_min(1e-3))
    self.act_scale.copy_(a.abs().mean(0).clamp_min(1e-3))
    if xi is not None:
      self.xi_scale.copy_(xi.abs().mean(0).clamp_min(1e-3))


class LearnedRewardHeads:
  """Drop-in ``heads`` wrapper: ``reward`` comes from the model, everything else delegates."""

  def __init__(self, base: Any, model: RewardModel, target_scale: float = 1.0) -> None:
    self.base = base
    self.model = model
    self.target_scale = float(target_scale)
    self.r_mean = 0.0
    self.r_std = 1.0
    self.clip_sigma = 5.0
    self.normalize = True
    # diagnostic term layout changes: the only imagined reward term is the learned one
    self.reward_term_names = ("learned_reward",)

  # ------------------------------------------------------------------ delegated hooks
  def __getattr__(self, name: str) -> Any:
    return getattr(self.base, name)

  @property
  def anchor_index(self) -> int:
    return self.base.anchor_index

  @property
  def reward_cfg(self) -> Any:
    return self.base.reward_cfg

  # ------------------------------------------------------------------ the reward itself
  def reward(self, s_next, action, prev_action, ref, bodies=None, xi=None):
    r = self.model(s_next, action, xi)
    if self.normalize:
      r = r * self.r_std + self.r_mean
      if self.clip_sigma > 0:
        lo = self.r_mean - self.clip_sigma * self.r_std
        hi = self.r_mean + self.clip_sigma * self.r_std
        r = r.clamp(lo, hi)
    return {"learned_reward": r, "total": r}

  def update(self, buffer, cfg: RewardModelCfg, optim: torch.optim.Optimizer) -> dict[str, float]:
    """Fit the model to the replay buffer's real rewards (MSE on normalised targets)."""
    self.normalize = bool(cfg.normalize)
    self.clip_sigma = float(cfg.clip_sigma)
    w = buffer.sample_windows(cfg.batch_size, 1)
    s, a, xi, target = w.states[0], w.a_exec[0], w.xi, w.reward[0]
    if cfg.normalize:
      # exponential moving statistics keep the regression well conditioned
      batch_mean = float(target.mean())
      batch_std = float(target.std().clamp_min(1e-6))
      self.r_mean = 0.9 * self.r_mean + 0.1 * batch_mean if self.r_std != 1.0 else batch_mean
      self.r_std = 0.9 * self.r_std + 0.1 * batch_std if self.r_std != 1.0 else batch_std
      self.r_std = max(self.r_std, 1e-6)
      y = (target - self.r_mean) / self.r_std
    else:
      y = target / self.target_scale
    loss = nn.functional.mse_loss(self.model(s, a, xi), y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    optim.step()
    with torch.no_grad():
      pred = self.model(s, a, xi)
      pred = pred * self.r_std + self.r_mean if cfg.normalize else pred * self.target_scale
      err = (pred - target).abs()
      corr = torch.corrcoef(torch.stack([pred, target]))[0, 1] if target.std() > 0 else torch.tensor(float("nan"))
    return {
      "reward_mse": float(loss),
      "reward_mae": float(err.mean()),
      "reward_corr": float(corr),
      "reward_mean": float(target.mean()),
    }
