"""Reward-contract equivalence: the spec path must equal the legacy analytic heads.

This is the test behind the README's ``3.7e-9`` claim. The refactor moved every reward term into a
reflected contract (``build_wm_spec``), so the one thing it must not do is change the training objective.
The check builds the engine's tracking environment, then evaluates both paths on identical states,
actions, references and context:

    python tests/test_reward_equivalence.py [--npz-dir DIR] [--num-envs 8]

It exits non-zero when any term, the observation frame, or a termination decision differs. Requires one
GPU and a motion directory with at least one clip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/tasks/supermimic/engine"))

import torch  # noqa: E402

REWARD_TERMS = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
  "action_rate_l2",
  "joint_limit",
  "total",
)
TOL = 1e-6


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz-dir", default="/home/tcs/8t01/humanoid/eewbc/motion_lib_1")
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  from src.tasks.supermimic.adapters.from_env import build_wm_spec
  from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter, build_g1_heads
  from src.tasks.supermimic.rl.config import SuperMimicCfg
  from supermimic.world_model.heads import analytic_observation
  from supermimic.world_model.wm_trainer import reference_at

  cfg = SuperMimicCfg(run_name="equiv", log_root="/tmp/equiv")
  cfg.data.npz_dir = args.npz_dir
  cfg.env.num_envs = args.num_envs
  cfg.env.anchor_obs = True
  cfg.device = args.device

  env = MotionTrackingAdapter.build_env(cfg)
  adapter = MotionTrackingAdapter(env)
  old_heads = build_g1_heads(env)
  spec = build_wm_spec(env, cfg)
  head_view = spec.build_heads(env)
  lib = spec.build_lib()

  env.reset()
  n = args.num_envs
  g = torch.Generator(device=args.device).manual_seed(0)
  frame = lib.clip_start[torch.zeros(n, dtype=torch.long, device=args.device)]
  ref = reference_at(lib, frame)

  s = adapter.state(env)
  a = torch.randn(n, 29, device=args.device, generator=g) * 0.3
  a_prev = torch.randn(n, 29, device=args.device, generator=g) * 0.3
  xi = adapter.xi(env)

  with torch.no_grad():
    old = old_heads.reward(s, a, a_prev, ref)
    new = spec.reward(s, a, a_prev, ref, xi=xi)
    o_old = analytic_observation(s, a_prev, xi, old_heads.default_joint_pos)
    o_new = head_view.observation(s, a_prev, xi, old_heads.default_joint_pos)

  print(f"{'term':26s} {'legacy':>14s} {'spec':>14s} {'max|diff|':>12s}")
  worst = 0.0
  for name in REWARD_TERMS:
    legacy_mean = float(old[name].mean())
    spec_mean = float(new[name].mean()) if name in new else float("nan")
    diff = float((old[name] - new[name]).abs().max()) if name in new else float("inf")
    worst = max(worst, diff)
    print(f"{name:26s} {legacy_mean:+14.8f} {spec_mean:+14.8f} {diff:12.3e}")

  obs_diff = float((o_old - o_new).abs().max())
  term_old = old_heads.termination_features(s[:, 0:3], s[:, 3:7], ref)
  term_new = spec.termination_features(s[:, 0:3], s[:, 3:7], ref)
  term_diff = float((term_old - term_new).abs().max())
  agrees = bool((old_heads.terminated(s, ref) == spec.terminated(s, ref)).all())

  print(f"\nobservation max|diff| {obs_diff:.3e}")
  print(f"termination features max|diff| {term_diff:.3e}")
  print(f"terminated agrees: {agrees}")

  ok = worst < TOL and obs_diff < TOL and term_diff < TOL and agrees
  print(f"\nWORST REWARD DIFF {worst:.3e}  ->  {'EQUIVALENT' if ok else 'MISMATCH'}")
  env.close()
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
