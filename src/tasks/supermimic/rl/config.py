"""Configuration for :class:`~src.tasks.supermimic.rl.runner.SuperMimicRunner`.

Mirrors the reference implementation's ``TrainConfig`` (same fields, same defaults) so a
run is controlled exactly like a PPO run: one dataclass tree, tyro/CLI friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from supermimic.config import (  # the vendored engine's config types
  DataCfg,
  EnvCfg,
  ScheduleCfg,
  TrainConfig,
)
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class SuperMimicCfg(TrainConfig):
  """``TrainConfig`` + what the generic runner needs (log dir, adapter name, ...).

  ``data``/``env`` are only used by the shipped adapters that build their own
  environment (``adapters/tracking.py``); a custom adapter ignores them.
  """

  #: which adapter to build when using ``scripts/train_supermimic.py``
  adapter: str = "tracking"
  #: mjlab-compatibility fields (read by scripts/train.py for any runner_cls)
  clip_actions: bool = True
  max_iterations: int = 20000
  experiment_name: str = ""
  wandb_tags: list[str] = field(default_factory=list)
  logger: str = "tensorboard"
  #: learn r(s, a, xi) from the replay buffer instead of hand-written analytic terms
  #: (mandatory for tasks whose reward needs contact forces)
  reward_model: bool = False
  #: refuse to start when a reward term cannot be evaluated in imagination (recommended)
  strict: bool = True
  #: terms without an analytic projection are learned from the env's per-term values
  learned_rewards: bool = True
  #: mjlab task id used by the "velocity" adapter
  task_id: str = "Mjlab-Velocity-Flat-Unitree-G1"
  #: override the policy/critic observation size (normally inferred from the adapter)
  obs_dim: int | None = None
  #: override the goal size (normally inferred from the adapter)
  goal_dim: int | None = None
  #: rollout horizon used by the imagined objective
  horizon: int = 32
  #: TF32 for the dense fp32 matmuls (Ampere/Ada tensor cores); free 1.3-2x here.
  tf32: bool = True
  #: number of world-model update groups to run *before* the first policy update
  policy_start_updates: int = 0
  #: update-to-data ratio (paper: 0.4)
  updates_per_env_step: float = 0.4
  #: learning rate for the world model / policy (paper: 2e-4)
  lr: float = 2e-4


__all__ = [
  "CriticCfg",
  "DataCfg",
  "EnvCfg",
  "PolicyCfg",
  "PolicyTrainerCfg",
  "ScheduleCfg",
  "SuperMimicCfg",
  "WMTrainerCfg",
  "WorldModelCfg",
]
