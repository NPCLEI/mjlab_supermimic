"""Task-side reward / termination changes must reach the *imagined* objective too.

The framework never needs editing for a task change: the environment gives the real
reward/termination for free, and the algorithm stays as it is. What a task change *does*
touch is the adapter, because the imagined rollout cannot call mjlab's reward functions
(they need contact forces / torques); it needs analytic, differentiable versions on the
**predicted** state.

So the split is:

* **framework** (`rl/runner.py`, `engine/`) -- never edited for a task change;
* **task env cfg** (`env_cfg.rewards`, `env_cfg.terminations`) -- free to edit, *and* this
  module keeps the analytic heads in sync automatically: reward weights, reward stds,
  dt and the termination thresholds are read back from the environment at construction;
* **adapter** -- edited only when you add a reward term that has no analytic
  implementation yet (implement it once in `heads.reward`).

This class is the second bullet: it subclasses the engine's ``AnalyticHeads`` and pulls
the numbers from ``env.cfg`` so that tuning the task does not silently desynchronise the
training objective from the scored objective.
"""

from __future__ import annotations

from typing import Any

import torch

from supermimic.world_model.heads import AnalyticHeads, TrackingRewardCfg

#: mjlab reward term name -> ``TrackingRewardCfg`` weight field
WEIGHT_FIELDS = {
  "motion_body_pos": "w_body_pos",
  "motion_body_ori": "w_body_ori",
  "motion_body_lin_vel": "w_body_lin_vel",
  "motion_body_ang_vel": "w_body_ang_vel",
  "motion_global_root_pos": "w_root_pos",
  "motion_global_root_ori": "w_root_ori",
  "action_rate_l2": "w_action_rate",
  "joint_limit": "w_joint_limit",
}

#: mjlab reward term name -> ``TrackingRewardCfg`` std field (read from the term params)
STD_FIELDS = {
  "motion_body_pos": "std_body_pos",
  "motion_body_ori": "std_body_ori",
  "motion_body_lin_vel": "std_body_lin_vel",
  "motion_body_ang_vel": "std_body_ang_vel",
}


class EnvSyncedHeads(AnalyticHeads):
  """``AnalyticHeads`` whose reward weights / stds / thresholds come from ``env.cfg``."""

  def __init__(self, env: Any, *args, **kwargs) -> None:
    base = kwargs.pop("reward_cfg", None) or TrackingRewardCfg()
    synced = self._sync_cfg(env, base)
    super().__init__(*args, reward_cfg=synced, **kwargs)
    self.termination_thresholds = self._read_thresholds(env)

  # ------------------------------------------------------------------ sync
  @staticmethod
  def _sync_cfg(env: Any, cfg: TrackingRewardCfg) -> TrackingRewardCfg:
    rewards = getattr(env.cfg, "rewards", None) or {}
    for name, field in WEIGHT_FIELDS.items():
      term = rewards.get(name)
      if term is not None and hasattr(term, "weight"):
        setattr(cfg, field, float(term.weight))
    for name, field in STD_FIELDS.items():
      term = rewards.get(name)
      params = getattr(term, "params", None) or {}
      if "std" in params:
        setattr(cfg, field, float(params["std"]))
    cfg.dt = float(getattr(env, "step_dt", cfg.dt))
    return cfg

  @staticmethod
  def _read_thresholds(env: Any) -> dict[str, float]:
    """Termination thresholds, if the task exposes them in the term params."""
    out: dict[str, float] = {}
    terms = getattr(env.cfg, "terminations", None) or {}
    for name, term in terms.items():
      params = getattr(term, "params", None) or {}
      if "threshold" in params:
        out[str(name)] = float(params["threshold"])
    return out

  # ------------------------------------------------------------------ termination
  def terminated(self, s: torch.Tensor, ref, height_thr: float | None = None, tilt_thr: float | None = None):
    """Same criterion as the environment, with the task's own thresholds by default."""
    if height_thr is None:
      height_thr = self._pick(("root_height", "anchor_pos", "bad_anchor_pos_z_only"), 0.25)
    if tilt_thr is None:
      tilt_thr = self._pick(("root_tilt", "anchor_ori", "bad_anchor_ori"), 0.8)
    return super().terminated(s, ref, height_thr=height_thr, tilt_thr=tilt_thr)

  def _pick(self, names: tuple[str, ...], default: float) -> float:
    for n in names:
      if n in self.termination_thresholds:
        return self.termination_thresholds[n]
    return default
