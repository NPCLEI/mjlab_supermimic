"""Register SuperMimic variants of the stock mjlab tasks -- the same registration shape.

Compare with the PPO tasks in this workspace:

    register_mjlab_task(
        task_id="Mjlab-SparseTrack-Unitree-G1",
        env_cfg=sparse_track_env_cfg(),
        play_env_cfg=sparse_track_env_cfg(play=True),
        rl_cfg=_runner("src.tasks...:SparseZModel", "g1_sparse_track_z"),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

The SuperMimic variant only swaps ``rl_cfg`` and ``runner_cls``; ``env_cfg`` is untouched.

Import this module for the tasks to become available:

    from src.tasks.supermimic.configs.example_tasks import *   # noqa: F401,F403
    python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1-SM --agent.max-iterations 20000
"""

from __future__ import annotations

from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from src.tasks.supermimic.registry import register_supermimic_task, supermimic_runner_cfg
from supermimic.world_model.state_spec import StateSpecCfg
from src.tasks.supermimic.rl.config import PolicyTrainerCfg, ScheduleCfg, WMTrainerCfg, WorldModelCfg


def _velocity_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_env_cfg()
  if play:
    cfg.scene.num_envs = 1
  return cfg


# --- velocity tracking, trained with SuperMimic instead of PPO -----------------------------
register_supermimic_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-SM",
  env_cfg=_velocity_env_cfg(),
  play_env_cfg=_velocity_env_cfg(play=True),
  rl_cfg=supermimic_runner_cfg(
    adapter="auto",
    reward_model=True,
    # sized for a 22 GB card shared with other jobs
    world_model=WorldModelCfg(
      state_spec=StateSpecCfg(
        # (root pos, root quat, joints, root lin vel, root ang vel, joint vel)
        err_scale=(0.05, 0.1, 0.5, 1.0, 1.0, 12.0),
        out_scale=(0.3, 0.5, 2.0, 0.01, 0.2, 0.5),
        in_scale=(1.0, 1.0, 1.0, 2.0, 10.0, 30.0),
      ),
    ),
    wm_trainer=WMTrainerCfg(batch_size=512, batches_per_update=4),
    policy_trainer=PolicyTrainerCfg(batch_size=1024, lr=1e-4, horizon=16),
    schedule=ScheduleCfg(warmup_iterations=200, total_iterations=200000),
    policy_start_updates=200,
  ),
)
