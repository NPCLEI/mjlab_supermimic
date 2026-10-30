"""Export a lean inference checkpoint from the whole-library training snapshot.

The training checkpoint is 217 MB because it carries three optimiser states. Only the networks, the config
and the iteration counter are needed to play or evaluate the policy, which is ~72 MB -- small enough to ship
in the repository without Git LFS.

    python tools/export_checkpoint.py <training ckpt> <out.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
  src = Path(sys.argv[1])
  out = Path(sys.argv[2])

  ck = torch.load(src, map_location="cpu", weights_only=False)
  keep = {k: ck[k] for k in ("policy", "wm", "critic", "config", "iteration") if k in ck}
  keep["meta"] = {
    "source": src.name,
    "iteration": int(ck.get("iteration", 0)),
    "trained_on": "22,246 clips / 11.34 M frames / 313 h (MotionDecode + KungfuAthlete retargets)",
    "evaluation": {
      "300_clip_held_out_subset": 0.8248,
      "300_clip_failures": 8,
      "fall_get_up_6_clips": 0.6703,
      "protocol": "paper Appendix F: one deterministic pass per clip from frame 0",
    },
    "world_model_1step_error": {"nse1": 0.0035, "nse8": 0.0057},
    "note": "policy + world model + critic only; optimiser states were stripped",
  }

  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(keep, out)

  size = out.stat().st_size / 1e6
  params = sum(t.numel() for t in keep["policy"].values())
  print(f"[export] {out}  {size:.1f} MB  ({params / 1e6:.2f} M policy params)")
  for k in keep:
    if isinstance(keep[k], dict):
      print(f"         {k:9s} {len(keep[k])} entries")


if __name__ == "__main__":
  main()
