"""Why is body_ori ~0 while body_ang_vel tracks at 85%?

If the orientation error is a *common* rotation shared by every tracked body (a systematic
offset, e.g. from the re-anchoring convention), that reward term is dead from the start and
no amount of training fixes it. If the errors differ per body, it is a genuine tracking
limitation.

Reports, over a live rollout: mean per-body orientation error, the pelvis (anchor) error,
and the residual after removing the pelvis rotation from every body.
"""

from __future__ import annotations

import argparse

import torch
from mjlab.utils.lab_api.math import quat_box_minus

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter, build_g1_heads
from src.tasks.supermimic.rl.config import SuperMimicCfg
from supermimic.envs.mdp.rewards import reanchor_reference
from supermimic.policy.nets import Policy


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default="logs/refimpl_one/cin1/ckpt_latest.pt")
  ap.add_argument("--data", default="motion_lib_1")
  ap.add_argument("--steps", type=int, default=300)
  args = ap.parse_args()

  cfg = SuperMimicCfg(run_name="ori", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = args.data
  cfg.env.num_envs = 64
  cfg.env.anchor_obs = True
  cfg.env.sampling_mode = "uniform"
  cfg.policy.goal_dim = cfg.env.future_frames * 68 + 9

  env = MotionTrackingAdapter.build_env(cfg)
  adapter = MotionTrackingAdapter(env)
  lib = adapter.build_lib()
  heads = build_g1_heads(env)
  ck = torch.load(args.ckpt, map_location=cfg.device, weights_only=False)
  policy = Policy(cfg.policy).to(cfg.device)
  policy.load_state_dict(ck["policy"])
  policy.eval()
  cmd = env.command_manager.get_term("motion")

  obs = env.reset()
  obs = obs[0] if isinstance(obs, tuple) else obs
  errs, resids, pelvis = [], [], []
  for _ in range(args.steps):
    with torch.no_grad():
      a = policy(obs["policy"], obs["goal"])
      s = adapter.state(env)
      frame = cmd.frame.clone()
      b = lib.bodies(frame)
      rb = heads.robot_bodies(s)
      ai = heads.anchor_index
      _, rel_quat = reanchor_reference(
        b.pos_w, b.quat_w,
        b.pos_w[..., ai, :], b.quat_w[..., ai, :],
        rb["pos"][..., ai, :], rb["quat"][..., ai, :],
      )
      d = quat_box_minus(rel_quat, rb["quat"])          # (N,B,3) rotation vector
      err = d.norm(dim=-1)                              # (N,B)
      errs.append(err.mean(0))
      pelvis.append(err[:, ai])
      resids.append((d - d[:, ai : ai + 1, :]).norm(dim=-1).mean(0))
    out = env.step(a)
    obs = out[0] if isinstance(out, tuple) else out

  e = torch.stack(errs).mean(0)
  p = torch.stack(pelvis).mean()
  r = torch.stack(resids).mean(0)
  print(f"mean per-body orientation error: {float(e.mean()):.3f} rad")
  print(f"pelvis (anchor) error:           {float(p):.3f} rad")
  print(f"residual after removing pelvis:  {float(r.mean()):.3f} rad")
  ratio = float(r.mean() / max(float(e.mean()), 1e-6))
  print(f"residual / total:                {ratio:.2f}  "
        f"({'mostly a COMMON rotation' if ratio < 0.5 else 'per-body, genuine tracking error'})")
  print("\nper body (worst first):")
  order = torch.argsort(e, descending=True).tolist()
  for idx in order[:8]:
    print(f"  body {idx:2d}  err {float(e[idx]):.3f}  residual {float(r[idx]):.3f}")
  env.close()


if __name__ == "__main__":
  main()
