"""Is the imagined reward the same as the real reward? (same data distribution)

Both sides are sampled from the *same* replay buffer of a live rollout, so the comparison
is like for like:

* **real**: H-step sums of the rewards the simulator actually produced;
* **imagined**: H-step sums the world model + analytic heads predict from the same kind
  of start states (the engine's own ``imagine``).

If the imagined return is much larger, the policy optimises the model's optimism (the
failure mode seen earlier); if it is close, the learning signal is trustworthy.

    env -u PYTHONPATH PYTHONPATH=. python -m src.tasks.supermimic.diag_imag_real \
        --ckpt logs/supermimic/one6/ckpt_0004000.pt --data motion_lib_1 --steps 400
"""

from __future__ import annotations

import argparse

import torch

import src.tasks.supermimic  # noqa: F401
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter, build_g1_heads
from src.tasks.supermimic.rl.config import SuperMimicCfg
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Policy
from supermimic.world_model.world_model import WorldModel


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default="logs/supermimic/one6/ckpt_latest.pt")
  ap.add_argument("--data", default="motion_lib_1")
  ap.add_argument("--steps", type=int, default=400)
  ap.add_argument("--horizon", type=int, default=16)
  ap.add_argument("--batch", type=int, default=256)
  args = ap.parse_args()

  cfg = SuperMimicCfg(run_name="diag", log_root="logs/supermimic")
  cfg.device = "cuda:0"
  cfg.data.npz_dir = args.data
  cfg.env.num_envs = 256
  cfg.env.anchor_obs = True
  cfg.env.sampling_mode = "uniform"
  cfg.policy.goal_dim = cfg.env.future_frames * 68 + 9
  cfg.wm_trainer.compile = None

  env = MotionTrackingAdapter.build_env(cfg)
  adapter = MotionTrackingAdapter(env)
  lib = adapter.build_lib()
  heads = build_g1_heads(env)
  ck = torch.load(args.ckpt, map_location=cfg.device, weights_only=False)

  policy = Policy(cfg.policy).to(cfg.device)
  policy.load_state_dict(ck["policy"])
  policy.eval()
  wm = WorldModel(cfg.world_model, heads, cfg.device)
  wm.load_state_dict(ck["wm"])
  wm.eval()

  obs = env.reset()
  obs = obs[0] if isinstance(obs, tuple) else obs
  for _ in range(args.steps):
    with torch.no_grad():
      a = policy(obs["policy"], obs["goal"])
    out = env.step(a)
    obs = out[0] if isinstance(out, tuple) else out
  buf = env.recorder_manager.get_term("replay").buffer
  print(f"buffer rows {buf.size}, horizon {args.horizon}", flush=True)

  w = buf.sample_windows(args.batch, args.horizon)
  real = w.reward.sum(0)

  start = buf.sample_starts(args.batch, cfg.policy.history)
  with torch.no_grad():
    roll = imagine(wm, policy, lib, start, args.horizon,
                   future_frames=cfg.env.future_frames, obs_noise=False, anchor_obs=True)
  imag = (roll.rewards * roll.survival[:-1]).sum(0)

  r_m, i_m = float(real.mean()), float(imag.mean())
  print(f"\nreal  H={args.horizon} return: mean {r_m:.4f}  p50 {float(real.median()):.4f}")
  print(f"imag  H={args.horizon} return: mean {i_m:.4f}  p50 {float(imag.median()):.4f}")
  print(f"ratio imag/real: {i_m / max(r_m, 1e-6):.3f}")
  print(f"|imag-real| mean {float((imag - real).abs().mean()):.4f}")
  print(f"imagined survival at H: {float(roll.survival[-1].mean()):.3f}")
  print("\nper-step imagined reward terms (mean over batch and horizon):")
  terms = roll.reward_terms
  if isinstance(terms, dict):
    for name, val in terms.items():
      print(f"  {name:26s} {float(val.mean()):+.4f}")
  else:
    for i, val in enumerate(terms):
      print(f"  term {i:22d} {float(val.mean() if val.dim() > 0 else val):+.4f}")
  env.close()


if __name__ == "__main__":
  main()
