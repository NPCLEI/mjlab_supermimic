"""Drop-in runner for ``register_mjlab_task(runner_cls=...)``.

mjlab's ``scripts/train.py`` builds the runner as

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device, **kwargs)
    if resume_path: runner.load(resume_path)
    runner.learn(num_learning_iterations=..., init_at_random_ep_len=True)

so a SuperMimic runner usable as ``runner_cls`` must accept the *wrapped* env, a plain dict
config, and that ``learn``/``load`` surface. This class is that adapter: it recovers the raw
mjlab env, rebuilds :class:`SuperMimicCfg` from the dict, picks a task adapter from the env
and delegates to :class:`SuperMimicRunner`.
"""

from __future__ import annotations

from typing import Any

from src.tasks.supermimic.adapters.auto import make_adapter
from src.tasks.supermimic.rl.config import SuperMimicCfg
from src.tasks.supermimic.rl.runner import SuperMimicRunner


def _cfg_from_dict(cfg):
  """Rebuild the nested SuperMimic config from what mjlab hands to a runner."""
  from src.tasks.supermimic.rl.cfg_roundtrip import cfg_from_dict

  return cfg_from_dict(cfg)


def _actor_history(env: Any) -> int:
  """History length of the env's actor group (0 when the terms are single-frame)."""
  mgr = getattr(env, "observation_manager", None)
  if mgr is None:
    return 1
  cfgs = getattr(mgr, "_group_obs_term_cfgs", {}).get("actor", [])
  return max([int(getattr(c, "history_length", 0) or 0) for c in cfgs] or [0])


class SuperMimicOnPolicyRunner:
  """``runner_cls`` compatible wrapper around :class:`SuperMimicRunner`."""

  def __init__(
    self,
    env: Any,
    cfg: dict[str, Any] | SuperMimicCfg,
    log_dir: str,
    device: str,
    registry_name: str | None = None,
    adapter: Any | None = None,
  ) -> None:
    raw = getattr(env, "unwrapped", env)
    sm_cfg = _cfg_from_dict(cfg)
    # a task whose actor group carries no history must not get the tracking history length
    if _actor_history(raw) == 0:
      sm_cfg.policy.history = 1
      sm_cfg.policy_trainer.history = 1
    if adapter is None:
      adapter = make_adapter(raw, sm_cfg)
    print(
      f"[INFO] SuperMimicOnPolicyRunner | adapter {type(adapter).__name__} | device {device} | log {log_dir}",
      flush=True,
    )
    self.impl = SuperMimicRunner(raw, sm_cfg, log_dir, device, adapter=adapter)
    self.device = device
    self.log_dir = log_dir

  # ------------------------------------------------------------------ mjlab runner surface
  def learn(self, num_learning_iterations: int | None = None, init_at_random_ep_len: bool = False) -> None:
    """Ignore PPO's ``num_learning_iterations``: SuperMimic is budgeted by wall-clock/schedule."""
    if num_learning_iterations is not None:
      self.impl.cfg.schedule.total_iterations = int(num_learning_iterations) * int(self.impl.env.num_envs)
    self.impl.learn()

  def load(self, path: str) -> None:
    self.impl.load(path)

  def save(self, path: str | None = None) -> None:
    self.impl.save(path)

  def add_git_repo_to_log(self, *args: Any, **kwargs: Any) -> None:
    """mjlab calls this right after construction; nothing to record here."""
    return None
