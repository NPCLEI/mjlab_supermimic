"""Render a 4x4 grid of the policy tracking 16 motions, into a GIF for the README.

Each clip is simulated **on its own** (one environment, one clip, stopped at its own last frame and then
held), because a shared 16-env rollout makes every cell reset when the shortest clip ends -- the grid
visibly "ends together" even though the motions did not. Rendering is done by replaying the recorded
``qpos`` in CPU MuJoCo, which avoids the Warp offscreen path and works headless.

    MUJOCO_GL=egl PYTHONPATH=.:src/tasks/supermimic/engine python scripts/render_grid.py \
        --ckpt <ckpt> --npz-dir <dir> --clips-file docs/assets/grid_clips.txt \
        --out docs/assets/grid.gif --steps 600 --stride 3 --cell 190 --colors 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

G1_XML = "/home/tcs/1t01/envs/capylab/lib/python3.10/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def clamp_to_limits(model, qpos: np.ndarray) -> np.ndarray:
  """Keep the replayed pose inside the model's hinge ranges.

  The policy's raw joint targets can sit outside a hinge limit for a frame or two; the simulator's
  constraint solver pulls them back, but replaying qpos directly would show a hyper-extended knee.
  Clamping makes the figure show the pose the simulator actually enforces.
  """
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
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--npz-dir", required=True)
  ap.add_argument("--out", default="docs/assets/grid.gif")
  ap.add_argument("--clips", type=int, default=16)
  ap.add_argument("--clips-file", default="", help="one 'path<TAB>label' per line; overrides the random pick")
  ap.add_argument("--steps", type=int, default=600, help="steps per clip; 50 Hz, so 600 = 12 s")
  ap.add_argument("--cell", type=int, default=190)
  ap.add_argument("--colors", type=int, default=48, help="GIF palette size; fewer colours means a smaller file")
  ap.add_argument("--stride", type=int, default=3, help="render every Nth simulated step; playback fps = 50/N")
  ap.add_argument("--fps", type=int, default=0, help="0 = derive from --stride so playback is real time")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  import mujoco
  from PIL import Image, ImageDraw

  from supermimic.envs import SuperMimicEnvCfgArgs, make_env
  from supermimic.envs.mdp.motion_library import discover_npz
  from supermimic.eval.benchmark import load_policy_from_checkpoint

  # ------------------------------------------------ the clip list
  labels: list[str] = []
  if args.clips_file:
    files: list[str] = []
    for line in Path(args.clips_file).read_text().splitlines():
      if not line.strip():
        continue
      parts = line.split("\t")
      files.append(parts[0].strip())
      labels.append(parts[1].strip() if len(parts) > 1 and parts[1].strip() else f"#{len(files)}")
  else:
    npz = discover_npz(args.npz_dir)
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(len(npz), size=min(args.clips, len(npz)), replace=False))
    files = [npz[i] for i in picks]
    labels = [f"#{i + 1}" for i in range(len(files))]

  policy, _cfg = load_policy_from_checkpoint(args.ckpt, args.device)
  policy.eval()

  # ------------------------------------------------ simulate one clip at a time
  n = len(files)
  trajs: list[np.ndarray] = []
  for i, f in enumerate(files):
    env_args = SuperMimicEnvCfgArgs(
      npz_files=[f],
      num_envs=1,
      future_frames=10,
      episode_length_s=1.0e5,   # the clip end decides; one clip never truncates another
      action_noise_std=0.0,
      obs_noise=False,
      reset_noise=False,
      domain_randomization=False,
      sampling_mode="start",
      record_transitions=False,
      seed=args.seed,
      anchor_obs=True,
    )
    env = make_env(env_args, device=args.device)
    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    cmd.forced_frame[:] = cmd.library.clip_start[:1]
    obs, _ = env.reset()

    traj = np.zeros((args.steps, 36), dtype=np.float32)
    ended = 0
    for t in range(args.steps):
      with torch.no_grad():
        act = policy(obs["policy"], obs["goal"])
      obs, _rew, term, _tout, _ = env.step(act)
      q = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                     robot.data.joint_pos], dim=-1)[0, :36]
      traj[t] = q.detach().cpu().numpy()
      if bool(term[0]):
        # clip finished (or the policy fell): hold the last frame rather than reset into a jump
        ended = t
        traj[t + 1 :] = traj[t]
        break
    env.close()
    trajs.append(traj)
    state = f"ended at {ended / 50:.1f}s" if ended else "ran the full window"
    print(f"[render] {i + 1:2d}/{n} {labels[i]:8s} {state}", flush=True)

  # ------------------------------------------------ CPU MuJoCo replay
  model = mujoco.MjModel.from_xml_path(G1_XML)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, args.cell, args.cell)
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 2.15, 130.0, -12.0
  opts = mujoco.MjvOption()
  opts.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

  ncols = 4
  nrows = int(np.ceil(n / ncols))
  tile = Image.new("RGB", (ncols * args.cell, nrows * args.cell), (18, 20, 24))
  frames = []
  for t in range(0, args.steps, args.stride):
    for i in range(n):
      data.qpos[:] = clamp_to_limits(model, trajs[i][t])
      data.qpos[3:7] /= max(np.linalg.norm(trajs[i][t, 3:7]), 1e-6)  # MuJoCo wants unit quaternions
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      root = data.qpos[:3]
      cam.lookat[:] = [root[0], root[1], root[2] * 0.9]
      renderer.update_scene(data, camera=cam, scene_option=opts)
      tile.paste(Image.fromarray(renderer.render()), ((i % ncols) * args.cell, (i // ncols) * args.cell))

    frame = tile.copy()
    draw = ImageDraw.Draw(frame)
    for i in range(n):
      x, y = (i % ncols) * args.cell + 6, (i // ncols) * args.cell + 4
      draw.text((x + 2, y + 2), labels[i], fill=(120, 200, 255))
    frames.append(frame)

  fps = args.fps or (50.0 / args.stride)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in frames]
  pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / fps),
              loop=0, optimize=True, disposal=1)
  print(f"[render] wrote {out} ({len(pal)} frames, {len(pal) / fps:.1f}s, "
        f"{out.stat().st_size / 1e6:.1f} MB)", flush=True)
  frames[len(frames) // 2].save(out.with_suffix(".png"))


if __name__ == "__main__":
  main()
