"""Pack the whole memmap motion library -- including the body arrays the dataset already has.

The engine's compact library only stored ``ref`` (root pose + joints) and recomputed the 14
tracked-body states with FK. The dataset already contains ``body_pos_w`` / ``body_quat_w`` /
``body_lin_vel_w`` / ``body_ang_vel_w`` for every frame, so store them too (fp16) and let
``bodies()`` gather them.

    python build_packed_library_full.py [out.pt] [max_clips]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "motions_all_sitfiltered"
ARRAYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def main() -> None:
  out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "motion_lib_all.pt"
  limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  idx = json.loads((SRC / "index.json").read_text())
  clips = idx["clips"] if not limit else idx["clips"][:limit]
  body_names = [str(b) for b in idx["body_names"]]
  root = body_names.index("pelvis")
  nb = len(body_names)
  print(f"{len(clips)} clips, {nb} bodies (root '{body_names[root]}' at {root})")

  mm = {k: np.load(SRC / f"{k}.npy", mmap_mode="r") for k in ARRAYS}
  lengths = np.array([int(c["end"]) - int(c["start"]) for c in clips], dtype=np.int64)
  starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
  total = int(lengths.sum())
  print(f"total frames {total} ({total / 50 / 3600:.2f} h @50Hz)")

  ref = np.empty((total, 71), dtype=np.float16)
  bpos = np.empty((total, nb, 3), dtype=np.float16)
  bquat = np.empty((total, nb, 4), dtype=np.float16)
  blin = np.empty((total, nb, 3), dtype=np.float16)
  bang = np.empty((total, nb, 3), dtype=np.float16)

  for i, c in enumerate(clips):
    s, e = int(c["start"]), int(c["end"])
    n = e - s
    w = int(starts[i])
    ref[w : w + n, 0:3] = mm["body_pos_w"][s:e, root, :]
    ref[w : w + n, 3:7] = mm["body_quat_w"][s:e, root, :]
    ref[w : w + n, 7:36] = mm["joint_pos"][s:e]
    ref[w : w + n, 36:39] = mm["body_lin_vel_w"][s:e, root, :]
    ref[w : w + n, 39:42] = mm["body_ang_vel_w"][s:e, root, :]
    ref[w : w + n, 42:71] = mm["joint_vel"][s:e]
    bpos[w : w + n] = mm["body_pos_w"][s:e]
    bquat[w : w + n] = mm["body_quat_w"][s:e]
    blin[w : w + n] = mm["body_lin_vel_w"][s:e]
    bang[w : w + n] = mm["body_ang_vel_w"][s:e]
    if (i + 1) % 4000 == 0:
      print(f"  {i + 1}/{len(clips)} clips", flush=True)

  torch.save(
    {
      "ref": torch.from_numpy(ref),
      "body_pos": torch.from_numpy(bpos),
      "body_quat": torch.from_numpy(bquat),
      "body_lin": torch.from_numpy(blin),
      "body_ang": torch.from_numpy(bang),
      "body_names": body_names,
      "names": [str(c["key"]) for c in clips],
      "starts": torch.from_numpy(starts),
      "lengths": torch.from_numpy(lengths),
    },
    out,
  )
  print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)  ref {tuple(ref.shape)}  bodies {tuple(bpos.shape)}")


if __name__ == "__main__":
  main()
