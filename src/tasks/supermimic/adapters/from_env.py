"""Build a :class:`WmSpec` from an mjlab environment config -- the only place a task is interpreted.

Everything the SuperMimic loop needs about a task is read off ``env`` / ``env.cfg`` here:

* ``env.cfg.rewards``      -> reward terms, classified analytic / learned / unavailable
* ``env.cfg.terminations`` -> the analytic part of the termination (time-outs are the horizon's job)
* ``env.observation_manager`` / ``command_manager`` -> observation and goal functions

A term whose *name* is registered in ``terms/registry.py`` is analytic; anything else becomes *learned*
(supervised by the per-term values the environment already logs), and a term with neither treatment is
dropped -- loudly, and fatally when ``strict`` is set. Nothing here names a specific task.
"""

from __future__ import annotations

from typing import Any

import torch

from src.tasks.supermimic.adapters.state import provider_for
from src.tasks.supermimic.adapters.terms.learned import LearnedRewardCfg, LearnedRewardGroup
from src.tasks.supermimic.adapters.terms.registry import bind_functions, family_of, resolve
from src.tasks.supermimic.rl.spec import RewardCfgShim, RewardCtx, SpecReport, WmTermSpec


def build_wm_spec(env: Any, cfg: Any) -> "WmSpec":
  """Reflect ``env`` into the training loop's single task contract."""
  provider = provider_for(env)
  dt = float(getattr(env, "step_dt", 0.02))
  heads = _build_heads(env)
  _register_families(heads)

  rewards = getattr(env.cfg, "rewards", {}) or {}
  bind_functions(rewards)

  reward_terms: list[WmTermSpec] = []
  analytic: list[str] = []
  learned: list[str] = []
  learned_weights: list[float] = []
  dropped: list[tuple[str, float]] = []

  for name, term_cfg in rewards.items():
    weight = float(getattr(term_cfg, "weight", 0.0))
    if weight == 0.0:
      continue
    func = getattr(term_cfg, "func", None)
    hit = resolve(func) if func is not None else None
    if hit is None:
      import importlib

      hit = resolve(importlib.import_module(hit.__class__.__name__)) if False else None
    if hit is not None:
      family, group, key = hit
      reward_terms.append(
        WmTermSpec(
          name=name,
          weight=weight,
          kind="analytic",
          fn=(lambda ctx, f=family, k=key, g=group: family_of(ctx, g, f)[k]),
          group=group,
        )
      )
      analytic.append(name)
    elif bool(getattr(cfg, "learned_rewards", True)):
      reward_terms.append(
        WmTermSpec(name=name, weight=weight, kind="learned", params=dict(getattr(term_cfg, "params", {}) or {}))
      )
      learned.append(name)
      learned_weights.append(weight)
    else:
      reward_terms.append(WmTermSpec(name=name, weight=weight, kind="unavailable"))
      dropped.append((name, weight))

  positives = sum(t.weight for t in reward_terms if t.weight > 0.0)
  max_step_reward = dt * positives

  spec = WmSpec(
    env=env,
    provider=provider,
    dt=dt,
    reward_terms=tuple(reward_terms),
    max_step_reward=max_step_reward,
    heads=heads,
  )

  if learned:
    spec.learned = LearnedRewardGroup(
      names=tuple(learned),
      weights=tuple(learned_weights),
      dt=dt,
      hidden=tuple(getattr(getattr(cfg, "learned_cfg", None), "hidden", (256, 256))),
    ).to(provider.state(env).device)

  spec.report = SpecReport(
    analytic=tuple(analytic),
    learned=tuple(learned),
    dropped=tuple(dropped),
    max_step_reward=max_step_reward,
  )
  print(spec.report.format(), flush=True)
  if dropped and bool(getattr(cfg, "strict", True)):
    raise RuntimeError(
      "strict=True: reward terms that cannot be evaluated in imagination were dropped: "
      + ", ".join(f"{n}({w:+.2f})" for n, w in dropped)
      + ". Provide an analytic projection in adapters/terms/registry.py or allow learned rewards."
    )
  return spec


# ---------------------------------------------------------------------- heads
def _build_heads(env: Any) -> Any:
  """Analytic heads for this env: the engine's synced heads for tracking, command heads otherwise."""
  from src.tasks.supermimic.adapters.state import _motion_term

  if _motion_term(env) is not None:
    from src.tasks.supermimic.adapters.tracking import build_g1_heads

    return build_g1_heads(env)

  from src.tasks.supermimic.adapters.velocity import VelocityHeads

  # the env's actor group plus the anchor features the adapter appends
  actor_dim = int(env.observation_manager.group_obs_dim["actor"][0])
  return VelocityHeads(env, actor_dim)


def _register_families(heads: Any) -> None:
  """Register the analytic families this task can use (idempotent, keyed by term name)."""
  if heads is None:
    return
  from src.tasks.supermimic.adapters.terms import families

  if heads.__class__.__name__ == "EnvSyncedHeads":
    families.register_tracking(heads)
  else:
    families.register_command(heads)


class WmSpec:
  """Uniform task surface consumed by ``SuperMimicRunner`` and the engine's imagination."""

  def __init__(self, *, env, provider, dt, reward_terms, max_step_reward, heads) -> None:
    self.env = env
    self.provider = provider
    self.dt = dt
    self.reward_terms = reward_terms
    self.max_step_reward = max_step_reward
    self.heads = heads
    self.learned: LearnedRewardGroup | None = None
    self.report: SpecReport | None = None
    self.reward_cfg = RewardCfgShim(max_step_reward, dt)
    self.action_bound = 1.0
    self.n_joints = getattr(provider, "n_joints", 29)
    self.anchor_index = int(getattr(heads, "anchor_index", 0))
    self.reward_term_names = tuple(t.name for t in reward_terms if t.kind in ("analytic", "learned"))

    # ---- runner contract flags, derived from the environment -------------------------------
    env_cfg = getattr(env, "cfg", None)
    self.auto_reset = bool(getattr(env_cfg, "auto_reset", True))
    has_replay_recorder = False
    recorder = getattr(env, "recorder_manager", None)
    if recorder is not None:
      try:
        has_replay_recorder = "replay" in list(getattr(recorder, "active_terms", []))
      except Exception:  # noqa: BLE001
        has_replay_recorder = False
    # the engine's tracking env records transitions itself; a stock mjlab task does not
    self.writes_buffer = not has_replay_recorder
    self._adds_action_noise = False

  def learned_columns(self, env=None) -> tuple[int, ...]:
    """Column indices of the learned terms inside ``env.reward_manager._step_reward``."""
    if self.learned is None or len(self.learned.names) == 0:
      return ()
    env = env or self.env
    names = list(getattr(env.reward_manager, "active_terms", []))
    return tuple(names.index(n) for n in self.learned.names if n in names)

  def drop_terms(self, names: tuple[str, ...] | list[str]) -> list[str]:
    """Remove reward terms from both the environment config and the imagined reward."""
    if not names:
      return []
    dropped: list[str] = []
    rewards = getattr(self.env.cfg, "rewards", {}) or {}
    for name in names:
      term_cfg = rewards.get(name)
      if term_cfg is not None:
        term_cfg.weight = 0.0  # the environment stops paying it too
      dropped.append(name)
    self.reward_terms = tuple(
      WmTermSpec(name=t.name, weight=t.weight, kind="unavailable", fn=None, group=t.group, params=t.params)
      if t.name in dropped
      else t
      for t in self.reward_terms
    )
    self.reward_term_names = tuple(n for n in self.reward_term_names if n not in dropped)
    positives = sum(t.weight for t in self.reward_terms if t.weight > 0.0)
    self.max_step_reward = self.dt * positives
    self.reward_cfg = RewardCfgShim(self.max_step_reward, self.dt)
    if self.learned is not None:
      keep = tuple(n for n in self.learned.names if n not in dropped)
      self.learned.names = keep
    return dropped

  # ------------------------------------------------------------------ state / context
  def state(self, env) -> torch.Tensor:
    return self.provider.state(env)

  def xi(self, env) -> torch.Tensor:
    return self.provider.xi(env)

  def frame_and_clip(self, env):
    return self.provider.frame_and_clip(env)

  # ------------------------------------------------------------------ reward
  def reward(self, s_next, action, prev_action, ref, bodies=None, xi=None):
    """Weighted, dt-scaled per-term values plus ``total`` -- the same numbers the environment reports."""
    ctx = RewardCtx(
      state=s_next,
      action=action,
      prev_action=prev_action,
      dt=self.dt,
      ref=ref,
      bodies=bodies,
      command=xi,
      default_joint_pos=getattr(self.heads, "default_joint_pos", None),
      soft_limits=getattr(self.heads, "soft_limits", None),
    )
    terms: dict[str, torch.Tensor] = {}
    for term in self.reward_terms:
      if term.kind == "analytic" and term.fn is not None:
        terms[term.name] = term.fn(ctx)
    if self.learned is not None and len(self.learned.names) > 0:
      terms.update({k: v for k, v in self.learned(s_next, action, xi).items() if k != "total"})
    total = None
    for name in self.reward_term_names:
      value = terms.get(name)
      if value is None:
        continue
      total = value if total is None else total + value
    terms["total"] = total if total is not None else torch.zeros(s_next.shape[0], device=s_next.device)
    return terms

  # ------------------------------------------------------------------ termination
  def termination_features(self, root_pos, root_quat, ref=None):
    if self.heads.__class__.__name__ == "EnvSyncedHeads" and ref is not None:
      return type(self.heads).termination_features(root_pos, root_quat, ref)
    from src.tasks.supermimic.adapters.velocity import VelocityHeads

    return VelocityHeads.termination_features(root_pos, root_quat, ref)

  def terminated(self, s, ref=None, height_thr=None, tilt_thr=None):
    if self.heads.__class__.__name__ == "EnvSyncedHeads" and ref is not None:
      return self.heads.terminated(s, ref, height_thr=height_thr, tilt_thr=tilt_thr)
    from src.tasks.supermimic.adapters.velocity import VelocityHeads

    return self.heads.terminated(s, ref, height_thr=height_thr, tilt_thr=tilt_thr) if hasattr(
      self.heads, "terminated"
    ) else VelocityHeads.terminated(self.heads, s, ref, height_thr, tilt_thr)

  # ------------------------------------------------------------------ observation / goal / bodies
  def robot_bodies(self, s):
    return self.heads.robot_bodies(s)

  def observation_frame(self, s_next, prev_action, xi, default_joint_pos):
    """Engine hook: one analytic observation frame (no history) from a predicted state."""
    if self.heads.__class__.__name__ == "EnvSyncedHeads":
      from supermimic.world_model.heads import analytic_observation

      return analytic_observation(s_next, prev_action, xi, default_joint_pos)
    return self.heads.observation(s_next, prev_action, xi, default_joint_pos)

  def observation(self, env) -> dict[str, torch.Tensor]:
    """Runner surface: the environment's observation dict plus the goal."""
    obs = env.observation_manager.compute()
    actor = obs["actor"] if "actor" in obs else obs["policy"]
    if self.heads.__class__.__name__ != "EnvSyncedHeads":
      # the actor group does not carry the base velocity; append the anchor-like error, exactly as
      # the imagination path does
      actor = torch.cat([actor, self.heads.anchor_feature(self.state(env), self.xi(env))], dim=-1)
    frames = int(getattr(self.env.cfg, "future_frames", 10) or 10)
    goal = self.goal_from_command(actor.shape[0], frames)
    return {"policy": actor, "goal": goal}

  def goal(self, start, ref, anchor_pose, future_frames):
    """Engine hook: per-sampled-row goal."""
    return self.goal_from_command_from_frame(start, int(future_frames))

  def goal_from_command(self, n: int, frames: int) -> torch.Tensor:
    from src.tasks.supermimic.adapters.state import _command

    cmd = _command(self.env)
    if cmd is None:
      raise RuntimeError("the task exposes no command term; cannot build a goal")
    return cmd[:n].repeat(1, frames)

  def goal_from_command_from_frame(self, start, frames: int) -> torch.Tensor:
    # per-row command carried in xi (see MjlabStateProvider.xi)
    return start.xi[..., 1:4].repeat(1, frames)

  def root_error_features(self, s, ref):
    if self.heads.__class__.__name__ == "EnvSyncedHeads":
      return type(self.heads).root_error_features(s, ref)
    from src.tasks.supermimic.adapters.velocity import VelocityHeads

    return VelocityHeads.root_error_features(s, ref)

  # ------------------------------------------------------------------ adapter-compat surface
  def build_heads(self, env):
    return _HeadView(self)

  def build_lib(self):
    if self.heads.__class__.__name__ == "EnvSyncedHeads":
      return self.env.command_manager.get_term("motion").library
    from src.tasks.supermimic.adapters.velocity import VelocityLib

    return VelocityLib(self.env)


class _HeadView:
  """Adapter between :class:`WmSpec` and the engine's ``heads`` protocol."""

  def __init__(self, spec: WmSpec) -> None:
    self._spec = spec

  def observation(self, s_next, prev_action, xi, default_joint_pos):
    return self._spec.observation_frame(s_next, prev_action, xi, default_joint_pos)

  def __getattr__(self, name):
    return getattr(self._spec, name)
