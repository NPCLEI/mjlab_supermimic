"""Single world-model update through the framework: is the loss finite?"""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="wmcheck", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 256
  cfg.policy_trainer.batch_size = 256
  cfg.wm_trainer.batch_size = 256
  cfg.schedule.total_iterations = 10**9
  cfg.schedule.warmup_iterations = 10**9
  cfg.policy.goal_dim = cfg.env.future_frames * 68
  cfg.wm_trainer.compile = None

  env = MotionTrackingAdapter.build_env(cfg)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/wmcheck", cfg.device,
                            adapter=MotionTrackingAdapter(env))
  obs = runner._observation()
  runner._on_reset(None)
  prev_s, prev_xi = runner._state(), runner._xi()
  ids = torch.arange(256, device=cfg.device)
  for _ in range(80):
    with torch.no_grad():
      a = runner.policy(obs["policy"], obs["goal"])
      a_env = runner._env_action(a) + 0.1 * torch.randn_like(a)
      obs, rew, term, tout, _ = env.step(a_env)
      done = term | tout
      runner.buffer.add(ids, s=prev_s, a=a, a_exec=a_env, xi=prev_xi, r=rew,
                        done=term.float(), frame=torch.zeros(256, dtype=torch.long, device=cfg.device),
                        clip=torch.zeros(256, dtype=torch.long, device=cfg.device),
                        terminal=torch.zeros_like(done), ep_start=torch.zeros_like(done))
      prev_s, prev_xi = runner._state(), runner._xi()
      if done.any():
        d = done.nonzero().squeeze(-1)
        env.reset(env_ids=d)
        runner._on_reset(d)
        obs = runner._observation()
        prev_s = runner._state()
  print(f"buffer size {runner.buffer.size}")
  for i in range(6):
    m = runner.wm_trainer.update(runner.buffer)
    fin = all(bool(torch.isfinite(p).all()) for p in runner.wm.parameters())
    print(f"  upd {i}: loss {m['loss']:.4f} nse1 {m['nse_1step']:.4f} "
          f"grad {m['grad_norm']:.2f} params_finite {fin}", flush=True)
  env.close()


if __name__ == "__main__":
  main()
