"""Pick a task adapter from the environment alone.

This is what makes ``runner_cls=SuperMimicRunner`` usable for any registered task: the runner
does not need a hand-written adapter per task, it inspects the env.

* the env has a motion command (``MotionCommand`` with a motion library) -> motion-tracking
  adapter, which provides the analytic body-tracking rewards the paper designed;
* otherwise -> generic adapter: 71-dim state from the robot, goal from the command manager,
  analytic rewards for whichever terms the task's own config exposes, tilt/height termination.

The generic path needs the reward to be a function of state (imagination has no simulator), so
contact-based terms are dropped and reported. A learned reward model (``cfg.reward_model``)
lifts that restriction.
"""

from __future__ import annotations

from typing import Any

from src.tasks.supermimic.rl.config import SuperMimicCfg


def _has_motion_command(env: Any) -> bool:
  mgr = getattr(env, "command_manager", None)
  if mgr is None:
    return False
  for name in getattr(mgr, "active_terms", []):
    term = mgr.get_term(name)
    for attr in ("motion", "motion_lib", "lib", "_motion"):
      if getattr(term, attr, None) is not None:
        return True
  return False


def make_adapter(env: Any, cfg: SuperMimicCfg | None = None) -> Any:
  """Return the task adapter for ``env`` (see the module docstring)."""
  which = getattr(cfg, "adapter", "auto") if cfg is not None else "auto"
  if which == "tracking":
    from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter

    return MotionTrackingAdapter(env)
  if which == "velocity":
    from src.tasks.supermimic.adapters.velocity import VelocityAdapter

    return VelocityAdapter(env)
  if which == "mjlab":
    from src.tasks.supermimic.adapters.mjlab_task import MjlabTrackingAdapter

    return MjlabTrackingAdapter(env)

  # auto
  if _has_motion_command(env):
    from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter

    print("[INFO] auto adapter: motion tracking (a motion command is present)", flush=True)
    return MotionTrackingAdapter(env)
  from src.tasks.supermimic.adapters.velocity import VelocityAdapter

  print("[INFO] auto adapter: generic command tracking (no motion library found)", flush=True)
  return VelocityAdapter(env)
