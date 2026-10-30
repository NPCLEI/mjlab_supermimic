"""Render the *reference* pose of a clip (no policy), to separate a bad retarget from a bad policy.

    python /tmp/ref_sheet.py <npz> <out.png> [steps]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

G1_XML = "/home/tcs/1t01/envs/capylab/lib/python3.10/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def clamp_to_limits(model, qpos):
  import mujoco

  out = qpos.copy()
  for j in range(model.njnt):
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    lo, hi = model.jnt_range[j]
    if hi > lo:
      a = model.jnt_qposadr[j]
      out[a] = min(max(out[a], lo), hi)
  return out


def main() -> None:
  npz, out = sys.argv[1], Path(sys.argv[2])
  steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600

  import mujoco
  from PIL import Image, ImageDraw

  d = np.load(npz)
  root_pos = d["body_pos_w"][:, 0]
  root_quat = d["body_quat_w"][:, 0]
  joint = d["joint_pos"]
  n = min(steps, joint.shape[0])
  qpos = np.concatenate([root_pos[:n], root_quat[:n], joint[:n]], axis=-1).astype(np.float32)
  print(f"reference: {n} frames ({n / 50:.1f}s) of {Path(npz).name[:50]}")

  model = mujoco.MjModel.from_xml_path(G1_XML)
  data = mujoco.MjData(model)
  cell = 190
  renderer = mujoco.Renderer(model, cell, cell)
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 2.15, 130.0, -12.0
  idx = np.linspace(0, n - 1, 12).astype(int)
  sheet = Image.new("RGB", (4 * cell, 3 * cell), (18, 20, 24))
  draw = ImageDraw.Draw(sheet)
  for k, t in enumerate(idx):
    data.qpos[:] = clamp_to_limits(model, qpos[t])
    data.qpos[3:7] /= max(np.linalg.norm(qpos[t, 3:7]), 1e-6)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.qpos[:3] * np.array([1, 1, 0.9])
    renderer.update_scene(data, camera=cam)
    sheet.paste(Image.fromarray(renderer.render()), ((k % 4) * cell, (k // 4) * cell))
    draw.text(((k % 4) * cell + 6, (k // 4) * cell + 4), f"t={t / 50:.1f}s", fill=(255, 190, 120))
  out.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(out)
  print("wrote", out)


if __name__ == "__main__":
  main()
