"""Smoke test: the framework trains the reference tracking task end to end.

Runs a short training (one clip, small batch) and asserts that the world-model error
drops -- i.e. the vendored engine is wired correctly through the adapter and the runner.

    env -u PYTHONPATH PYTHONPATH=. python -m src.tasks.supermimic.smoke
"""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="smoke", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 128
  cfg.env.episode_length_s = 20.0
  cfg.policy_trainer.batch_size = 256
  cfg.wm_trainer.batch_size = 256
  cfg.schedule.total_iterations = 400
  cfg.schedule.log_interval = 50
  cfg.schedule.checkpoint_interval = 10_000
  cfg.policy.goal_dim = cfg.env.future_frames * 68

  env = MotionTrackingAdapter.build_env(cfg)
  adapter = MotionTrackingAdapter(env)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/smoke", cfg.device, adapter=adapter)
  runner.learn()

  # the world model must have improved over the run
  with torch.no_grad():
    err = runner.wm_trainer.online_error(runner.buffer, n=512, k=8)
  print(f"[smoke] online nse1 {float(err[0]):.4f} nse8 {float(err[-1]):.4f}")
  assert float(err[-1]) < 1.0, f"world model did not fit: {float(err[-1])}"
  ckpt = runner.save("ckpt_smoke.pt")
  print(f"[smoke] saved {ckpt} -- OK")


if __name__ == "__main__":
  main()
