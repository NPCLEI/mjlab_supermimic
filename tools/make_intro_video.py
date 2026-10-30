"""Build the project introduction video (1280x720, H.264) from the measured figures.

Structure: title -> what the method is -> the task contract -> four result slides -> the 16-motion grid
(rendered at video resolution) -> the velocity negative result -> limitations and credits. English headline
with a Chinese one-line gloss on each slide, matching the bilingual README.

    MUJOCO_GL=egl PYTHONPATH=.:src/tasks/supermimic/engine python tools/make_intro_video.py \
        --ckpt <ckpt> --npz-dir <dir> --clips-file docs/assets/grid_clips.txt --out docs/assets/intro.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (14, 17, 22)
FG = (232, 236, 242)
DIM = (150, 162, 178)
ACCENT = (96, 180, 255)
WARM = (255, 176, 96)
GOOD = (120, 220, 160)

TITLE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BODY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
  try:
    return ImageFont.truetype(path, size)
  except Exception:  # noqa: BLE001
    return ImageFont.load_default()


def blank() -> Image.Image:
  return Image.new("RGB", (W, H), BG)


def wrap(text: str, fnt, width: int) -> list[str]:
  words, lines, cur = text.split(), [], ""
  for w in words:
    trial = (cur + " " + w).strip()
    if fnt.getlength(trial) <= width:
      cur = trial
    else:
      lines.append(cur)
      cur = w
  if cur:
    lines.append(cur)
  return lines


def text_slide(title: str, zh: str, bullets: list[tuple[str, tuple[int, int, int]]],
               footer: str = "", kicker: str = "") -> Image.Image:
  img = blank()
  d = ImageDraw.Draw(img)
  y = 78
  if kicker:
    d.text((80, 54), kicker, font=font(BODY_FONT, 20), fill=DIM)
    y = 96
  d.text((80, y), title, font=font(TITLE_FONT, 46), fill=FG)
  y += 66
  d.text((80, y), zh, font=font(CJK_FONT, 24), fill=ACCENT)
  y += 62
  for text, colour in bullets:
    d.rectangle([80, y + 9, 86, y + 15], fill=colour)
    for line in wrap(text, font(BODY_FONT, 25), W - 200):
      d.text((102, y), line, font=font(BODY_FONT, 25), fill=FG)
      y += 34
    y += 14
  if footer:
    d.text((80, H - 62), footer, font=font(BODY_FONT, 19), fill=DIM)
  return img


def code_slide() -> Image.Image:
  img = blank()
  d = ImageDraw.Draw(img)
  d.text((80, 54), "THE EXTENSION POINT", font=font(BODY_FONT, 20), fill=DIM)
  d.text((80, 92), "A task is a registration, not a subclass", font=font(TITLE_FONT, 40), fill=FG)
  d.text((80, 148), "新增任务只需一次注册，训练器不知道任务存在", font=font(CJK_FONT, 23), fill=ACCENT)

  code = [
    ("register_mjlab_task(", WARM),
    ("    task_id=...,", FG),
    ("    env_cfg=tracking_env_cfg(),      # rewards, terminations, obs", DIM),
    ("    play_env_cfg=tracking_env_cfg(play=True),", FG),
    ("    rl_cfg=supermimic_runner_cfg(    # training hyper-parameters only", DIM),
    ("        world_model=WorldModelCfg(input_norm=True),", FG),
    ("        policy_trainer=PolicyTrainerCfg(horizon=32),", FG),
    ("        strict=True,                 # refuse unevaluable rewards", DIM),
    ("    ),", FG),
    ("    runner_cls=SuperMimicOnPolicyRunner,", FG),
    (")", WARM),
  ]
  mono = font(MONO_FONT, 20)
  y = 208
  for line, colour in code:
    d.text((80, y), line, font=mono, fill=colour)
    y += 27

  d.text((80, H - 108), "Reward and termination are reflected from env_cfg, then verified:",
         font=font(BODY_FONT, 21), fill=FG)
  d.text((80, H - 76), "tests/test_reward_equivalence.py   worst term difference 3.7e-9, exit code gated",
         font=font(MONO_FONT, 20), fill=GOOD)
  return img


def figure_slide(png: Path, title: str, zh: str, note: str) -> Image.Image:
  img = blank()
  d = ImageDraw.Draw(img)
  d.text((80, 52), title, font=font(TITLE_FONT, 38), fill=FG)
  d.text((80, 104), zh, font=font(CJK_FONT, 22), fill=ACCENT)
  fig = Image.open(png).convert("RGB")
  fig = fig.crop((0, 34, fig.width, fig.height))  # drop the figure's own title; the slide has one
  scale = min((W - 160) / fig.width, (H - 250) / fig.height)
  fig = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
  img.paste(fig, ((W - fig.width) // 2, 150))
  d.text((80, H - 56), note, font=font(BODY_FONT, 20), fill=DIM)
  return img


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--npz-dir", required=True)
  ap.add_argument("--clips-file", default="docs/assets/grid_clips.txt")
  ap.add_argument("--out", default="docs/assets/intro.mp4")
  ap.add_argument("--grid-seconds", type=float, default=12.0)
  ap.add_argument("--grid-gif", default="", help="reuse an already rendered grid gif")
  args = ap.parse_args()

  root = Path(__file__).resolve().parents[1]
  assets = root / "docs/assets"
  frames_dir = Path("/tmp/intro_frames")
  if frames_dir.exists():
    shutil.rmtree(frames_dir)
  frames_dir.mkdir(parents=True)

  timeline: list[tuple[Image.Image, int]] = []  # (image, frames)

  # 1. title
  t = blank()
  d = ImageDraw.Draw(t)
  d.text((80, 210), "SuperMimic in mjlab", font=font(TITLE_FONT, 72), fill=FG)
  d.text((80, 300), "one differentiable world model, one policy, 22,246 motions", font=font(BODY_FONT, 30), fill=ACCENT)
  d.text((80, 352), "用一个可微世界模型训练 Unitree G1 的全身动作跟踪策略", font=font(CJK_FONT, 26), fill=DIM)
  d.text((80, 470), "clip 1307  0.849 return, zero failures     300-clip subset  0.8248",
         font=font(MONO_FONT, 22), fill=GOOD)
  d.text((80, H - 70), "github.com/NPCLEI/mjlab_supermimic     arXiv:2610.28480 (CoRL 2026)",
         font=font(BODY_FONT, 19), fill=DIM)
  timeline.append((t, int(4.5 * FPS)))

  # 2. what it is
  timeline.append((text_slide(
    "What this is", "方法：世界模型 + 想象中反传，不需要信任域",
    [("A 71-dim state world model s_{t+1} = f(s_t, a_t, xi) over qpos|qvel, trained on replayed transitions.", FG),
     ("A tracking policy trained by backpropagating through 32 imagined steps, with a critic bootstrapping the tail.", FG),
     ("Differentiable reward and termination heads, so the imagined objective is the scored objective.", FG)],
    footer="Measured on one RTX 4090 and one A100; every number uses the paper's Appendix-F protocol.",
    kicker="METHOD"), int(7 * FPS)))

  # 3. the contract
  timeline.append((code_slide(), int(9 * FPS)))

  # 4-7. results
  timeline.append((figure_slide(assets / "fig3_whole_library.png", "Whole library: one model beats a specialist",
                                "全量库：一个世界模型，通用策略优于逐条专用模型",
                                "22,246 clips / 313 h retargeted from MotionDecode and KungfuAthlete"),
                   int(7 * FPS)))
  timeline.append((figure_slide(assets / "fig1_1307_benchmark.png", "Single motion: 0.849 with zero failures",
                                "单条动作：315.8 秒片段跑完，零失败",
                                "clip 1307, 15,792 / 15,792 steps, input normalisation reaches the return sooner"),
                   int(7 * FPS)))
  timeline.append((figure_slide(assets / "fig2_world_model_error.png", "World model: normalisation buys 2.4x",
                                "输入归一化让世界模型误差在 3k 迭代时低 2.4 倍",
                                "running mean/variance on state, action and xi, updated from real transitions only"),
                   int(6 * FPS)))
  timeline.append((figure_slide(assets / "fig5_learned_rewards.png", "Which rewards can be learned",
                                "哪些奖励项能被学出来：只有一项过 0.9 闸门",
                                "the rest depend on contact and phase, so they are removed from the reward on both sides"),
                   int(6 * FPS)))

  # 8. the 16-motion grid, rendered at video resolution by the grid script
  grid_dir = Path("/tmp/intro_grid")
  if not args.grid_gif:
    if grid_dir.exists():
      shutil.rmtree(grid_dir)
    grid_dir.mkdir(parents=True)
  if args.grid_gif:
    shutil.copy(args.grid_gif, grid_dir / "grid.gif")
  else:
    subprocess.run([
      sys.executable, str(root / "scripts/render_grid.py"),
      "--ckpt", args.ckpt, "--npz-dir", args.npz_dir, "--clips-file", args.clips_file,
      "--out", str(grid_dir / "grid.gif"), "--steps", str(int(args.grid_seconds * 50)),
      "--stride", "2", "--cell", "300", "--colors", "64",
    ], check=True)

  # the grid script writes a mid-frame PNG; rebuild the animation frames from its GIF
  gif = Image.open(grid_dir / "grid.gif")
  gif_frames = []
  try:
    while True:
      gif_frames.append(gif.convert("RGB"))
      gif.seek(gif.tell() + 1)
  except EOFError:
    pass
  print(f"[video] grid frames: {len(gif_frames)}")

  left, top = 44, 128
  side = H - top - 44
  legend_x = left + side + 42
  cats = ["sprint", "jog", "jump", "martial", "crawl", "climb", "carry", "ground",
          "crouch", "sit", "itch", "dance", "turn", "walk", "idle", "other"]
  for gf in gif_frames:
    canvas = blank()
    d = ImageDraw.Draw(canvas)
    d.text((left, 44), "16 kinds of motion, one policy", font=font(TITLE_FONT, 34), fill=FG)
    d.text((left, 86), "同一策略，16 种不同类别动作，每格 12 秒", font=font(CJK_FONT, 20), fill=ACCENT)
    canvas.paste(gf.resize((side, side), Image.LANCZOS), (left, top))
    for k, name in enumerate(cats):
      col, row = divmod(k, 8)
      d.text((legend_x + col * 150, top + 44 + row * 34), f"#{k + 1}  {name}",
             font=font(BODY_FONT, 21), fill=FG)
    d.text((legend_x, top + 340), "each cell:", font=font(BODY_FONT, 20), fill=DIM)
    d.text((legend_x, top + 368), "the most dynamic clip", font=font(BODY_FONT, 20), fill=DIM)
    d.text((legend_x, top + 392), "of its category", font=font(BODY_FONT, 20), fill=DIM)
    timeline.append((canvas, 1))

  # 9. velocity negative result
  timeline.append((figure_slide(assets / "fig4_velocity.png", "What does not work: velocity tracking",
                                "没做出来的部分：速度追踪（4 种配置全部 0/64）",
                                "measured ceiling: world-model action sensitivity is 11.6 % of the simulator's"),
                   int(8 * FPS)))

  # 10. limitations + credits
  timeline.append((text_slide(
    "Limitations and credit", "局限与致谢",
    [("Fall/get-up is the weak class (0.6703, 6/6 failures); the paper closes it with 9.8 h on two GPUs.", WARM),
     ("Velocity tracking plateaus at 6-10 % of commands, with the reason measured rather than guessed.", WARM),
     ("BONES-SEED was not used, so failure counts are not directly comparable to the paper's.", DIM),
     ("Method and protocol: Liao, Shao, Long, Zakka, Horowitz, Abbeel and Sreenath, SuperMimic, CoRL 2026.", GOOD),
     ("Motion data: MotionDecode (CHINGMU / EnDecodeX) and KungfuAthlete (arXiv:2602.13656).", GOOD)],
    footer="Independent re-implementation, not affiliated with or endorsed by the authors.",
    kicker="THANKS"), int(9 * FPS)))

  # ------------------------------------------------ write frames
  idx = 0
  for img, count in timeline:
    for _ in range(count):
      img.save(frames_dir / f"{idx:05d}.jpg", quality=92)
      idx += 1
  print(f"[video] wrote {idx} frames ({idx / FPS:.1f}s)")

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", str(frames_dir / "%05d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)]
  subprocess.run(cmd, check=True)
  print(f"[video] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
