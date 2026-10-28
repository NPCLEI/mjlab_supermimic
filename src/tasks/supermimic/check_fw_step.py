"""Step the vendored world model by hand on a sampled window and locate the NaN."""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg
from supermimic.world_model.heads import ReferenceFrame


def main() -> None:
  cfg = SuperMimicCfg(run_name="stepcheck", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 64
  cfg.schedule.total_iterations = 10**9
  cfg.schedule.warmup_iterations = 10**9
  cfg.policy.goal_dim = cfg.env.future_frames * 68
  cfg.wm_trainer.compile = None

  env = MotionTrackingAdapter.build_env(cfg)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/stepcheck", cfg.device,
                            adapter=MotionTrackingAdapter(env))
  obs = runner._observation()
  runner._on_reset(None)
  prev_s, prev_xi = runner._state(), runner._xi()
  ids = torch.arange(64, device=cfg.device)
  for _ in range(60):
    with torch.no_grad():
      a = runner.policy(obs["policy"], obs["goal"])
      a_env = a + 0.1 * torch.randn_like(a)
      obs, rew, term, tout, _ = env.step(a_env)
      done = term | tout
      runner.buffer.add(ids, s=prev_s, a=a, a_exec=a_env, xi=prev_xi, r=rew,
                        done=term.float(), frame=torch.zeros(64, dtype=torch.long, device=cfg.device),
                        clip=torch.zeros(64, dtype=torch.long, device=cfg.device),
                        terminal=torch.zeros_like(done), ep_start=torch.zeros_like(done))
      prev_s, prev_xi = runner._state(), runner._xi()
      if done.any():
        d = done.nonzero().squeeze(-1)
        env.reset(env_ids=d)
        runner._on_reset(d)
        obs = runner._observation()
        prev_s = runner._state()

  w = runner.buffer.sample_windows(256, 8)
  print(f"window states {tuple(w.states.shape)} a_exec {tuple(w.a_exec.shape)} xi {tuple(w.xi.shape)}")
  print(f"a_exec finite {bool(torch.isfinite(w.a_exec).all())} xi finite {bool(torch.isfinite(w.xi).all())}")
  # per-step rollout
  s = w.states[0]
  k = w.a_exec.shape[0]
  for j in range(k):
    s = runner.wm.step(s, w.a_exec[j], w.xi)
    e = runner.wm.spec.error_vector(s, w.states[j + 1])
    r = runner.lib.ref(w.frame[j].reshape(-1))
    feats = runner.wm.heads.termination_features(s[..., 0:3], s[..., 3:7],
                                                 ReferenceFrame(None, None, None, None, r.root_pos_w, r.root_quat_w))
    print(f"  step {j}: s finite {bool(torch.isfinite(s).all())} "
          f"err finite {bool(torch.isfinite(e).all())} feats finite {bool(torch.isfinite(feats).all())} "
          f"|s|max {float(s.abs().max()):.3e}")
    if not bool(torch.isfinite(s).all()):
      bad = (~torch.isfinite(s)).nonzero()[:3]
      print("    first bad indices", bad.tolist())
      break
  env.close()


if __name__ == "__main__":
  main()
