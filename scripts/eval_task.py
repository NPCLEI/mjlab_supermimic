#!/usr/bin/env python
"""Appendix-F evaluation: replay every clip of a library once, report the normalised return.

  python scripts/eval_task.py --ckpt logs/run/ckpt_0018000.pt --npz-dir motion_lib_1 --num-envs 1
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/tasks/supermimic/engine"))

import tyro  # noqa: E402

from supermimic.envs.mdp.motion_library import discover_npz  # noqa: E402
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize  # noqa: E402


def main(ckpt: Path, npz_dir: Path = Path("motion_lib_1"), num_envs: int = 1, device: str = "cuda:0", out: Path | None = None) -> None:
  files = [str(f) for f in discover_npz(npz_dir)]
  policy, cfg = load_policy_from_checkpoint(ckpt, device)
  results = run_benchmark(policy, files, device=device, num_envs=num_envs,
                          future_frames=cfg["env"]["future_frames"],
                          anchor_obs=cfg["env"].get("anchor_obs", False))
  s = summarize(results)
  print(f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f}")
  for k, v in s.get("terms", {}).items():
    print(f"  {k:28s} {v:.4f}")
  save_results(results, out or ckpt.parent / f"benchmark_{ckpt.stem}.json")


if __name__ == "__main__":
  tyro.cli(main)
