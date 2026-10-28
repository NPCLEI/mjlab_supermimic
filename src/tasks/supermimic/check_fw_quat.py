"""Locate the zero-quaternion rows in the replay buffer."""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="quat", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 256
  cfg.schedule.total_iterations = 10**9
  cfg.schedule.warmup_iterations = 10**9
  cfg.policy.goal_dim = cfg.env.future_frames * 68
  cfg.wm_trainer.compile = None

  env = MotionTrackingAdapter.build_env(cfg)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/quat", cfg.device,
                            adapter=MotionTrackingAdapter(env))
  b = runner.buffer
  obs = runner._observation()
  runner._on_reset(None)
  prev_s, prev_xi = runner._state(), runner._xi()
  ids = torch.arange(256, device=cfg.device)
  for _ in range(80):
    with torch.no_grad():
      a = runner.policy(obs["policy"], obs["goal"])
      a_env = a + 0.1 * torch.randn_like(a)
      obs, rew, term, tout, _ = env.step(a_env)
      done = term | tout
      b.add(ids, s=prev_s, a=a, a_exec=a_env, xi=prev_xi, r=rew, done=term.float(),
            frame=torch.zeros(256, dtype=torch.long, device=cfg.device),
            clip=torch.zeros(256, dtype=torch.long, device=cfg.device),
            terminal=torch.zeros_like(done), ep_start=torch.zeros_like(done))
      prev_s, prev_xi = runner._state(), runner._xi()
      if done.any():
        d = done.nonzero().squeeze(-1)
        s_next = runner._state()
        zeros_a = torch.zeros(d.numel(), 29, device=cfg.device)
        b.add(d, s=s_next[d], a=zeros_a, a_exec=zeros_a, xi=runner._xi()[d],
              r=torch.zeros(d.numel(), device=cfg.device), done=term[d].float(),
              frame=torch.zeros(d.numel(), dtype=torch.long, device=cfg.device),
              clip=torch.zeros(d.numel(), dtype=torch.long, device=cfg.device),
              terminal=torch.ones(d.numel(), dtype=torch.bool, device=cfg.device),
              ep_start=torch.zeros(d.numel(), dtype=torch.bool, device=cfg.device))
        b.end_episode(d)
        env.reset(env_ids=d)
        runner._on_reset(d)
        obs = runner._observation()
        prev_s = runner._state()

  qn = b.s[..., 3:7].norm(dim=-1)          # (E,T)
  bad = (b.valid) & (qn < 1e-6)
  print(f"valid rows {int(b.valid.sum())}, zero-quat valid rows {int(bad.sum())}")
  idx = bad.nonzero()[:5]
  for e, t in idx.tolist():
    print(f"  env {e} t {t}: valid {bool(b.valid[e, t])} terminal {bool(b.terminal[e, t])} "
          f"ep_start {bool(b.ep_start[e, t])} ep_id {int(b.ep_id[e, t])} ptr {int(b.ptr[e])} "
          f"qnorm {float(qn[e, t]):.3e}")
  if len(idx):
    e, t = idx[0].tolist()
    print(f"  state row: {b.s[e, t].tolist()[:10]}")
  env.close()


if __name__ == "__main__":
  main()
