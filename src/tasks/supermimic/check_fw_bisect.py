"""Bisect the NaN across world-model batches: which sample breaks the loss?"""

from __future__ import annotations

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic import SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter
from src.tasks.supermimic.rl.config import SuperMimicCfg


def main() -> None:
  cfg = SuperMimicCfg(run_name="bisect", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = "src/tasks/supermimic_tracking/refimpl/clips"
  cfg.data.limit = 1
  cfg.env.num_envs = 256
  cfg.schedule.total_iterations = 10**9
  cfg.schedule.warmup_iterations = 10**9
  cfg.policy.goal_dim = cfg.env.future_frames * 68
  cfg.wm_trainer.compile = None

  env = MotionTrackingAdapter.build_env(cfg)
  runner = SuperMimicRunner(env, cfg, "logs/supermimic/bisect", cfg.device,
                            adapter=MotionTrackingAdapter(env))
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

  for i in range(10):
    w = runner.buffer.sample_windows(2048, 8)
    loss, m = runner.wm_trainer.loss_on(w)
    nan = not bool(torch.isfinite(loss))
    print(f"batch {i}: loss {float(loss):.4f} nse1 {float(m['nse_1step']):.4f} "
          f"bce {float(m['term_bce']):.4f} nan {nan}", flush=True)
    if nan:
      st = w.states
      print(f"  states finite {bool(torch.isfinite(st).all())} |s|max {float(st.abs().max()):.3e}")
      print(f"  quat min norm {float(st[..., 3:7].norm(dim=-1).min()):.3e}")
      print(f"  ang_vel max {float(st[..., 39:42].abs().max()):.3e} "
            f"lin_vel max {float(st[..., 36:39].abs().max()):.3e} "
            f"joint_vel max {float(st[..., 42:71].abs().max()):.3e}")
      print(f"  a_exec max {float(w.a_exec.abs().max()):.3e} xi max {float(w.xi.abs().max()):.3e}")
      print(f"  done [min,max] {float(w.done.min())},{float(w.done.max())}")
      # per-step error to find the first diverging step
      s = st[0]
      for j in range(8):
        s = runner.wm.step(s, w.a_exec[j], w.xi)
        e = runner.wm.spec.error_vector(s, st[j + 1])
        print(f"   step {j}: s finite {bool(torch.isfinite(s).all())} "
              f"err finite {bool(torch.isfinite(e).all())} |s|max {float(s.abs().max()):.3e}")
        if not bool(torch.isfinite(e).all()):
          bad = (~torch.isfinite(e)).nonzero()[:3]
          print("    bad err idx", bad.tolist())
          break
      break
  env.close()


if __name__ == "__main__":
  main()
