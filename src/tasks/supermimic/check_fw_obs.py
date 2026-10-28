"""Is mjlab's observation manager stateful? (does compute() advance history buffers?)"""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="obscheck", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 32
  cfg.policy.goal_dim = cfg.env.future_frames * 68
  env = MotionTrackingAdapter.build_env(cfg)
  env.reset()
  a = torch.zeros(32, 29, device="cuda:0")
  raw = env.step(a)          # returns whatever the env returns
  obs_step = raw[0]
  print("step-returned obs type:", type(obs_step).__name__,
        list(obs_step) if isinstance(obs_step, dict) else tuple(obs_step.shape))
  m = env.observation_manager
  c1 = m.compute()
  c2 = m.compute()
  k = "policy"
  d = float((c1[k] - c2[k]).abs().max()) if isinstance(c1, dict) else float("nan")
  print(f"two consecutive compute() calls differ by max {d:.3e}")
  if isinstance(obs_step, dict) and k in obs_step:
    d2 = float((obs_step[k] - c1[k]).abs().max())
    print(f"step-returned obs vs compute(): max diff {d2:.3e}")
  else:
    for i, t in enumerate(obs_step if isinstance(obs_step, (tuple, list)) else [obs_step]):
      if isinstance(t, torch.Tensor) and t.shape[-1] == 93:
        print(f"tensor #{i} shape {tuple(t.shape)} vs compute policy 93-dim: "
              f"max diff {float((t - c1[k]).abs().max()):.3e}")
  env.close()


if __name__ == "__main__":
  main()
