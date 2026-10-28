"""Build a multi-clip npz library for the reference SuperMimic engine.

The repo's ``motions_all_sitfiltered`` is a memmap library (6 ``.npy`` arrays +
``index.json`` whose ``body_names`` / ``joint_names`` we can reuse verbatim). The
reference implementation trains N clips at once from one npz per clip, so each clip is
sliced out of the memmap by its ``[start, end)`` range -- no conversion, no retargeting,
the arrays are already the mjlab ``MotionLoader`` layout at 50 Hz.

    python build_npz_library.py [n_clips] [out_dir]

Writes ``<out>/<safe_name>.npz`` with the six arrays plus ``body_names``.
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


def safe(name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace("/", "__"))[:120]


def main(n: int = 300, out: str | None = None) -> None:
  idx = json.loads((SRC / "index.json").read_text())
  clips = idx["clips"]
  body_names = np.array(idx["body_names"])
  print(f"library: {len(clips)} clips, {len(body_names)} tracked bodies: {list(body_names)}")
  print(f"joint_names: {len(idx['joint_names'])}")

  out_dir = Path(out) if out else REPO / f"motion_lib_{n}"
  out_dir.mkdir(parents=True, exist_ok=True)
  existing = len(list(out_dir.glob("*.npz")))

  # spread the selection over the library so the subset is not one category
  step = max(len(clips) // n, 1)
  picked = clips[::step][:n]
  print(f"picking {len(picked)} clips (every {step}-th), out={out_dir}, existing={existing}")

  mm = {k: np.load(SRC / f"{k}.npy", mmap_mode="r") for k in ARRAYS}
  frames = 0
  for i, clip in enumerate(picked):
    path = out_dir / f"{safe(clip['key'])}.npz"
    if path.exists():
      continue
    s, e = int(clip["start"]), int(clip["end"])
    np.savez(
      path,
      body_names=body_names,
      joint_pos=np.asarray(mm["joint_pos"][s:e], dtype=np.float32),
      joint_vel=np.asarray(mm["joint_vel"][s:e], dtype=np.float32),
      body_pos_w=np.asarray(mm["body_pos_w"][s:e], dtype=np.float32),
      body_quat_w=np.asarray(mm["body_quat_w"][s:e], dtype=np.float32),
      body_lin_vel_w=np.asarray(mm["body_lin_vel_w"][s:e], dtype=np.float32),
      body_ang_vel_w=np.asarray(mm["body_ang_vel_w"][s:e], dtype=np.float32),
    )
    frames += e - s
    if i % 50 == 0:
      print(f"  {i}/{len(picked)} ... ({frames} frames)", flush=True)

  total = len(list(out_dir.glob("*.npz")))
  print(f"done: {total} npz in {out_dir}"
        f"  ({sum(len(np.load(p if isinstance(p, Path) else p)['joint_pos']) for p in list(out_dir.glob('*.npz'))[:1])} frames sampled)")
  print(f"PATH {out_dir}")


if __name__ == "__main__":
  main(int(sys.argv[1]) if len(sys.argv) > 1 else 300,
       sys.argv[2] if len(sys.argv) > 2 else None)
