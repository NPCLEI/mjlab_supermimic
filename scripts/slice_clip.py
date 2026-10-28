"""Build a single-clip npz library, picking the longest clip in the memmap index.

    python build_one_clip.py [out_dir] [clip_key_substring]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "motions_all_sitfiltered"
ARRAYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main() -> None:
  out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "motion_lib_1"
  want = sys.argv[2] if len(sys.argv) > 2 else None
  idx = json.loads((SRC / "index.json").read_text())
  clips = idx["clips"]
  pool = [c for c in clips if want and want in c["key"]] or clips
  clip = max(pool, key=lambda c: int(c["n_frames"]))
  print(f"picked {clip['key']}  {clip['n_frames']} frames ({clip['n_frames'] / 50:.1f} s) "
        f"category {clip.get('category')}")

  out.mkdir(parents=True, exist_ok=True)
  s, e = int(clip["start"]), int(clip["end"])
  mm = {k: np.load(SRC / f"{k}.npy", mmap_mode="r") for k in ARRAYS}
  name = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip["key"].replace("/", "__"))[:120]
  path = out / f"{name}.npz"
  np.savez(
    path,
    body_names=np.array(idx["body_names"]),
    joint_pos=np.asarray(mm["joint_pos"][s:e], dtype=np.float32),
    joint_vel=np.asarray(mm["joint_vel"][s:e], dtype=np.float32),
    body_pos_w=np.asarray(mm["body_pos_w"][s:e], dtype=np.float32),
    body_quat_w=np.asarray(mm["body_quat_w"][s:e], dtype=np.float32),
    body_lin_vel_w=np.asarray(mm["body_lin_vel_w"][s:e], dtype=np.float32),
    body_ang_vel_w=np.asarray(mm["body_ang_vel_w"][s:e], dtype=np.float32),
  )
  print(f"wrote {path}")


if __name__ == "__main__":
  main()
