"""Does tuning the task's reward/termination flow into the imagined objective?

Changes a reward weight and a termination threshold on the environment config *only*
(no framework code, no adapter code), then checks that the analytic heads pick them up.
"""

from __future__ import annotations

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter, build_g1_heads
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="synccheck", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 8
  env = MotionTrackingAdapter.build_env(cfg)

  heads = build_g1_heads(env)
  print("before:", {k: v for k, v in vars(heads.reward_cfg).items() if k.startswith("w_")})
  print("thresholds before:", heads.termination_thresholds)

  # ---- edit ONLY the task config -------------------------------------------
  name = "motion_body_pos"
  env.cfg.rewards[name].weight = 3.5
  if "action_rate_l2" in env.cfg.rewards:
    env.cfg.rewards["action_rate_l2"].weight = -0.7
  for tname in ("root_height", "root_tilt"):
    if tname in env.cfg.terminations and "threshold" in env.cfg.terminations[tname].params:
      env.cfg.terminations[tname].params["threshold"] = 0.42

  heads2 = build_g1_heads(env)
  print("after: ", {k: v for k, v in vars(heads2.reward_cfg).items() if k.startswith("w_")})
  print("thresholds after: ", heads2.termination_thresholds)
  assert abs(heads2.reward_cfg.w_body_pos - 3.5) < 1e-9, "reward weight did not sync"
  assert abs(heads2.termination_thresholds.get("root_height", -1) - 0.42) < 1e-9, "threshold did not sync"
  print("[ok] the analytic heads follow the task config automatically")
  env.close()


if __name__ == "__main__":
  main()
