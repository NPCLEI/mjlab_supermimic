"""Render a 4x4 grid of the generalist policy tracking 16 motions, into a GIF for the README.

The policy runs in the same environment the benchmark uses (MuJoCo Warp); rendering is done by replaying
the recorded ``qpos`` in CPU MuJoCo, which avoids the Warp offscreen path and works headless. One GIF,
16 clips, 3 s each, camera following the robot.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_grid.py \
        --ckpt <whole-library ckpt> --npz-dir <motion_lib_300> --out docs/assets/grid.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

G1_XML = "/home/tcs/1t01/envs/capylab/lib/python3.10/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--npz-dir", required=True)
  ap.add_argument("--out", default="docs/assets/grid.gif")
  ap.add_argument("--clips", type=int, default=16)
  ap.add_argument("--steps", type=int, default=150, help="150 steps = 3 s at 50 Hz")
  ap.add_argument("--cell", type=int, default=240)
  ap.add_argument("--colors", type=int, default=96, help="GIF palette size; fewer colours means a smaller file")
  ap.add_argument("--fps", type=int, default=0, help="0 = derive from --stride so playback is real time")
  ap.add_argument("--stride", type=int, default=2, help="render every Nth simulated step")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--clips-file", default="", help="one 'path\\tlabel' per line; overrides the random pick")
  args = ap.parse_args()

  import mujoco
  from PIL import Image, ImageDraw

  from supermimic.envs import SuperMimicEnvCfgArgs, make_env
  from supermimic.envs.mdp.motion_library import discover_npz
  from supermimic.eval.benchmark import load_policy_from_checkpoint

  labels: list[str] = []
  if args.clips_file:
    for line in Path(args.clips_file).read_text().splitlines():
      if not line.strip():
        continue
      parts = line.split("\t")
      labels.append(parts[1].strip() if len(parts) > 1 else f"#{len(labels) + 1}")
      # the listed paths stay in order
      Path(args.clips_file)
    files = [line.split("\t")[0].strip() for line in Path(args.clips_file).read_text().splitlines() if line.strip()]
    npz = files
  else:
    npz = discover_npz(args.npz_dir)
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(len(npz), size=min(args.clips, len(npz)), replace=False))
    files = [npz[i] for i in picks]

  env_args = SuperMimicEnvCfgArgs(
    npz_files=files,
    num_envs=len(files),
    future_frames=10,
    episode_length_s=1.0e5,
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
  lib = cmd.library
  robot = env.scene["robot"]

  policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
  policy.eval()

  n = len(files)
  cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=args.device)]

  def short_name(path: str, index: int) -> str:
    # prefer the category label from --clips-file; otherwise the retarget id is unreadable, so number it
    return labels[index] if index < len(labels) else f"#{index + 1}"

  names = [short_name(f, i) for i, f in enumerate(files)]
  obs, _ = env.reset()

  # qpos (n, 36) per step, in the engine state layout
  traj = np.zeros((args.steps, n, 36), dtype=np.float32)

  for t in range(args.steps):
    with torch.no_grad():
      act = policy(obs["policy"], obs["goal"])
    obs, rew, term, tout, _ = env.step(act)
    q = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                   robot.data.joint_pos], dim=-1)[:, :36]
    traj[t] = q.detach().cpu().numpy()
  env.close()

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
      data.qpos[:] = traj[t, i]
      data.qpos[3:7] /= max(np.linalg.norm(traj[t, i, 3:7]), 1e-6)  # MuJoCo wants unit quats
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      root = data.qpos[:3]
      cam.lookat[:] = [root[0], root[1], root[2] * 0.9]
      renderer.update_scene(data, camera=cam, scene_option=opts)
      img = Image.fromarray(renderer.render())
      tile.paste(img, ((i % ncols) * args.cell, (i // ncols) * args.cell))

    frame = tile.copy()
    draw = ImageDraw.Draw(frame)
    for i in range(n):
      x, y = (i % ncols) * args.cell + 6, (i // ncols) * args.cell + 4
      draw.text((x + 2, y + 2), names[i], fill=(120, 200, 255))
    frames.append(frame)

  fps = args.fps or (50.0 / args.stride)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  # GIF is palettised; fewer colours keeps a 10 s 4x4 grid inside a sane file size
  pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in frames]
  pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / fps),
              loop=0, optimize=True, disposal=1)
  print(f"[render] wrote {out} ({len(pal)} frames, {len(pal) / fps:.1f}s, "
        f"{out.stat().st_size / 1e6:.1f} MB)", flush=True)
  frames[len(frames) // 2].save(out.with_suffix(".png"))


if __name__ == "__main__":
  main()
