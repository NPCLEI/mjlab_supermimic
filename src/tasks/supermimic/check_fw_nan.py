"""Where does the framework run's NaN come from?

Steps the tracking env through the runner's collection path and reports non-finite
counts per replay-buffer field, plus the same for the reward/state coming out of the
simulator.
"""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="nancheck", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 64
  cfg.policy_trainer.batch_size = 64
  cfg.wm_trainer.batch_size = 64
  cfg.schedule.total_iterations = 10**9
  cfg.schedule.warmup_iterations = 10**9  # collect only
  cfg.policy.goal_dim = cfg.env.future_frames * 68

  env = MotionTrackingAdapter.build_env(cfg)
  adapter = MotionTrackingAdapter(env)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/nancheck", cfg.device, adapter=adapter)

  obs = runner._observation()
  runner._on_reset(None)
  s = runner._state()
  print(f"initial state finite: {bool(torch.isfinite(s).all())}", flush=True)

  bad_steps = 0
  for i in range(60):
    with torch.no_grad():
      a = runner.policy(obs["policy"], obs["goal"])
      a_env = runner._env_action(a) + 0.1 * torch.randn_like(a)
      obs, rew, term, tout, _ = env.step(a_env)
      s2 = runner._state()
      if not bool(torch.isfinite(rew).all()) or not bool(torch.isfinite(s2).all()):
        bad_steps += 1
        if bad_steps <= 3:
          print(f"  step {i}: rew finite {bool(torch.isfinite(rew).all())} "
                f"state finite {bool(torch.isfinite(s2).all())} "
                f"max|rew| {float(rew.abs().max()):.3e}", flush=True)
      done = term | tout
      if done.any():
        env.reset(env_ids=done.nonzero().squeeze(-1))
        runner._on_reset(done.nonzero().squeeze(-1))
        obs = runner._observation()
  print(f"bad steps: {bad_steps}/60")

  # refill the buffer through the runner path, then inspect it
  runner.buffer = None
  from supermimic.data.replay_buffer import ReplayBuffer

  runner.buffer = ReplayBuffer(64, cfg.env.replay_capacity, 71, 29, 33, cfg.device)
  obs = runner._observation()
  runner._on_reset(None)
  prev_s = runner._state()
  prev_xi = runner._xi()
  for i in range(40):
    with torch.no_grad():
      a = runner.policy(obs["policy"], obs["goal"])
      a_env = runner._env_action(a) + 0.1 * torch.randn_like(a)
      obs, rew, term, tout, _ = env.step(a_env)
      done = term | tout
      runner.buffer.add(
        torch.arange(64, device=cfg.device), s=prev_s, a=a, a_exec=a_env, xi=prev_xi,
        r=rew, done=term.float(), frame=torch.zeros(64, dtype=torch.long, device=cfg.device),
        clip=torch.zeros(64, dtype=torch.long, device=cfg.device),
        terminal=torch.zeros_like(done), ep_start=torch.zeros_like(done),
      )
      prev_s = runner._state()
      prev_xi = runner._xi()
      if done.any():
        ids = done.nonzero().squeeze(-1)
        env.reset(env_ids=ids)
        runner._on_reset(ids)
        obs = runner._observation()
        prev_s = runner._state()
  for field in ("s", "a", "a_exec", "xi", "r", "done", "frame"):
    t = getattr(runner.buffer, field)
    n_bad = int((~torch.isfinite(t.float())).sum())
    print(f"  buffer.{field:8s} non-finite {n_bad}/{t.numel()}")
  w = runner.buffer.sample_windows(64, 8)
  for field in ("states", "a_exec", "xi", "reward", "done"):
    t = getattr(w, field)
    print(f"  window.{field:8s} non-finite {int((~torch.isfinite(t.float())).sum())}/{t.numel()}")
  env.close()


if __name__ == "__main__":
  main()
