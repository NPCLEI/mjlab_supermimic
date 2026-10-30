"""Contact sheet of one clip: 12 frames sampled across its playback, to spot bad poses.

    python /tmp/debug_clip.py <ckpt> <npz> <out.png> [label]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

G1_XML = "/home/tcs/1t01/envs/capylab/lib/python3.10/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def clamp_to_limits(model, qpos: np.ndarray) -> np.ndarray:
  """Keep the replayed pose inside the model's hinge ranges, so a solver-limited joint does not render as
  a hyper-extended knee."""
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
  ckpt, npz, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
  label = sys.argv[4] if len(sys.argv) > 4 else Path(npz).name[:40]
  steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600

  import mujoco
  from PIL import Image, ImageDraw

  from supermimic.envs import SuperMimicEnvCfgArgs, make_env
  from supermimic.eval.benchmark import load_policy_from_checkpoint

  policy, _ = load_policy_from_checkpoint(ckpt, "cuda:0")
  policy.eval()

  env = make_env(
    SuperMimicEnvCfgArgs(
      npz_files=[npz], num_envs=1, future_frames=10, episode_length_s=1.0e5,
      action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
      sampling_mode="start", record_transitions=False, seed=0, anchor_obs=True,
    ),
    device="cuda:0",
  )
  cmd = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  cmd.forced_frame[:] = cmd.library.clip_start[:1]
  obs, _ = env.reset()

  traj = np.zeros((steps, 36), dtype=np.float32)
  ended = 0
  for t in range(steps):
    with torch.no_grad():
      act = policy(obs["policy"], obs["goal"])
    obs, _r, term, _to, _ = env.step(act)
    traj[t] = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                         robot.data.joint_pos], dim=-1)[0, :36].detach().cpu().numpy()
    if bool(term[0]):
      ended = t
      traj[t + 1 :] = traj[t]
      break
  env.close()
  print(f"clip {Path(npz).name}: {ended if ended else steps} steps ({'ended' if ended else 'full'})")

  model = mujoco.MjModel.from_xml_path(G1_XML)
  data = mujoco.MjData(model)
  cell = 190
  renderer = mujoco.Renderer(model, cell, cell)
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 2.15, 130.0, -12.0
  idx = np.linspace(0, max(ended, 1) - 1, 12).astype(int) if ended else np.linspace(0, steps - 1, 12).astype(int)

  sheet = Image.new("RGB", (4 * cell, 3 * cell), (18, 20, 24))
  draw = ImageDraw.Draw(sheet)
  for k, t in enumerate(idx):
    data.qpos[:] = clamp_to_limits(model, traj[t])
    data.qpos[3:7] /= max(np.linalg.norm(traj[t, 3:7]), 1e-6)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.qpos[:3] * np.array([1, 1, 0.9])
    renderer.update_scene(data, camera=cam)
    sheet.paste(Image.fromarray(renderer.render()), ((k % 4) * cell, (k // 4) * cell))
    draw.text(((k % 4) * cell + 6, (k // 4) * cell + 4), f"t={t / 50:.1f}s", fill=(120, 200, 255))
  out.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(out)
  print("wrote", out)


if __name__ == "__main__":
  main()
