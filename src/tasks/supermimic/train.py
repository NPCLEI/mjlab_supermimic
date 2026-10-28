"""Train any task with SuperMimic -- the equivalent of PPO's ``train.py``.

    python -m src.tasks.supermimic.train \
        --adapter tracking \
        --data.npz-dir motion_lib_300 \
        --env.num-envs 2048 --schedule.max-hours 3.0 \
        --log-root logs/supermimic --run-name demo

The adapter (``tracking`` / ``mjlab``) is the only task-specific switch; everything else
is the same dataclass tree as the reference implementation.

TF32 is on by default: the bottleneck is dense fp32 matmuls (8.8M-param MLP with
2048-wide layers), where Ampere/Ada tensor cores give a free 1.3-2x on the same silicon.
Pass ``--env.no-tf32`` for plain fp32.
"""

from __future__ import annotations

import torch
import tyro

import src.tasks.supermimic  # noqa: F401  (installs the vendored engine on sys.path)
from src.tasks.supermimic.rl.config import SuperMimicCfg
from src.tasks.supermimic.rl.runner import SuperMimicRunner


def main(cfg: SuperMimicCfg) -> None:
  if cfg.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] TF32 enabled (matmul + cudnn)", flush=True)

  if cfg.adapter == "tracking":
    cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
  if cfg.adapter == "tracking":
    from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter

    env = MotionTrackingAdapter.build_env(cfg)
    adapter = MotionTrackingAdapter(env)
  elif cfg.adapter == "velocity":
    from src.tasks.supermimic.adapters.velocity import VelocityAdapter

    # the stock velocity task has no observation history and no reference motion
    cfg.policy.history = 1
    cfg.policy_trainer.history = 1
    env = VelocityAdapter.build_env(cfg)
    adapter = VelocityAdapter(env)
    print(f"[INFO] velocity adapter on {cfg.task_id} | actor dim {adapter._heads.actor_dim}", flush=True)
  elif cfg.adapter == "mjlab":
    raise SystemExit(
      "the 'mjlab' adapter needs a task id: build the env yourself and pass it to\n"
      "SuperMimicRunner(env, cfg, log_dir, device, adapter=MjlabTrackingAdapter(env))"
    )
  else:
    raise SystemExit(f"unknown adapter {cfg.adapter!r} (expected 'tracking' or 'mjlab')")

  log_dir = f"{cfg.log_root}/{cfg.run_name}"
  runner = SuperMimicRunner(env, cfg, log_dir, cfg.device, adapter=adapter)
  print(f"[INFO] {env.num_envs} envs | log_dir {log_dir}", flush=True)
  runner.learn()


if __name__ == "__main__":
  main(tyro.cli(SuperMimicCfg))
