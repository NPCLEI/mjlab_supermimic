"""``register_supermimic_task`` -- the same registration shape as the PPO tasks.

    from src.tasks.supermimic.registry import register_supermimic_task, supermimic_runner_cfg
    from src.tasks.velocity.configs.env_cfg import velocity_env_cfg

    register_supermimic_task(
        task_id="Mjlab-Velocity-Flat-Unitree-G1-SM",
        env_cfg=velocity_env_cfg(),                 # unchanged mjlab task config
        play_env_cfg=velocity_env_cfg(play=True),
        rl_cfg=supermimic_runner_cfg(adapter="auto", schedule=SchCfg(max_hours=8.0)),
        # runner_cls defaults to SuperMimicOnPolicyRunner
    )

and then ``python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1-SM`` trains it with
SuperMimic exactly like the PPO tasks -- the only thing that changes is ``rl_cfg`` and
``runner_cls``.
"""

from __future__ import annotations

from typing import Any

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.supermimic.rl.config import SuperMimicCfg
from src.tasks.supermimic.rl.mjlab_runner import SuperMimicOnPolicyRunner


def supermimic_runner_cfg(**overrides: Any) -> SuperMimicCfg:
  """Build the ``rl_cfg`` for a SuperMimic task (an analogue of ``_runner(...)``)."""
  cfg = SuperMimicCfg(**overrides)
  # pipeline defaults that differ from the PPO runner
  cfg.tf32 = True
  return cfg


def register_supermimic_task(
  task_id: str,
  env_cfg: Any,
  play_env_cfg: Any,
  rl_cfg: SuperMimicCfg | None = None,
  runner_cls: type | None = None,
) -> None:
  """Register ``task_id`` so ``scripts/train.py <task_id>`` trains it with SuperMimic."""
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_env_cfg,
    rl_cfg=rl_cfg or supermimic_runner_cfg(),
    runner_cls=runner_cls or SuperMimicOnPolicyRunner,
  )
