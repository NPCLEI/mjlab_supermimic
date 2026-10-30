"""Pick 16 clips that are both *large-amplitude* and *different in kind*.

Ranking by a single amplitude score returns sixteen sprints, so each clip is first assigned a category
from its name and the grid takes the most dynamic clip of each category. Amplitude is measured from the
stored reference data, not guessed from the name: peak root speed (finite differences of the root body,
50 Hz), root height range, and the total range of joint motion.

    python scripts/pick_grid_clips.py --npz-dir <dir> --count 16 --out /tmp/grid_clips.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DT = 1.0 / 50.0  # the retargeted library is 50 Hz

# first match wins; the specific words come before the generic ones
CATEGORIES = (
  ("sprint", ("sprint", "run_away", "run_ff", "panic_run", "titan_run", "_run_")),
  ("jog", ("jog",)),
  ("jump", ("jump", "leap", "hop")),
  ("martial", ("kungfu", "kung_fu", "punch", "kick", "fight", "martial", "box")),
  ("acrobatic", ("cartwheel", "flip", "handstand", "tumble", "hurdle", "spin")),
  ("crawl", ("crawl",)),
  ("climb", ("ladder", "climb")),
  ("carry", ("lift", "crate", "carry", "throw", "catch", "chainsaw")),
  ("throw", ("throw", "catch", "ball")),
  ("labor", ("chainsaw", "lumber", "shovel", "dig", "hammer", "cutting")),
  ("ground", ("ground", "floor", "faint", "stand_up", "get_up", "getup", "lie", "roll")),
  ("crouch", ("crouch", "squat", "kneel")),
  ("sit", ("sitting", "sit_", "sit__", "chair", "stool")),
  ("itch", ("itch", "scratch", "rub")),
  ("dance", ("dance", "chaines")),
  ("turn", ("turn", "twist", "pivot")),
  ("walk", ("walk", "sideway", "strafe", "backward")),
  ("idle", ("idle", "stand", "breath", "sit")),
)


def category_of(name: str) -> str:
  low = name.lower()
  for label, keys in CATEGORIES:
    if any(k in low for k in keys):
      return label
  return "other"


def root_index(body_names) -> int:
  names = [str(b).lower() for b in body_names]
  for want in ("pelvis", "root", "torso", "hip"):
    for i, n in enumerate(names):
      if want in n:
        return i
  return 0


def metrics(path: Path) -> dict[str, float] | None:
  try:
    d = np.load(path)
    body_pos = d["body_pos_w"]
    joint = d["joint_pos"]
    names = d["body_names"]
  except Exception:  # noqa: BLE001
    return None
  if body_pos.shape[0] < 40:
    return None
  root = body_pos[:, root_index(names)]
  vel = np.diff(root[:, :2], axis=0) / DT
  speed = float(np.linalg.norm(vel, axis=-1).max())
  height = float(root[:, 2].max() - root[:, 2].min())
  joint_range = float((np.percentile(joint, 97, axis=0) - np.percentile(joint, 3, axis=0)).sum())
  return {"speed": speed, "height": height, "joint": joint_range, "frames": float(body_pos.shape[0])}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz-dir", required=True)
  ap.add_argument("--count", type=int, default=16)
  ap.add_argument("--min-frames", type=int, default=120)
  ap.add_argument("--per-category", type=int, default=2, help="only used when there are fewer categories than slots")
  ap.add_argument("--out", default="/tmp/grid_clips.txt")
  args = ap.parse_args()

  rows = []
  for f in sorted(Path(args.npz_dir).glob("*.npz")):
    m = metrics(f)
    if m is None or m["frames"] < args.min_frames:
      continue
    rows.append({"path": str(f), "cat": category_of(f.name), **m})
  if not rows:
    raise SystemExit("no eligible clips")

  for key, weight in (("speed", 0.45), ("height", 0.30), ("joint", 0.25)):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
    span = max(hi - lo, 1e-6)
    for r in rows:
      r[f"n_{key}"] = float(np.clip((r[key] - lo) / span, 0.0, 1.0))
    for r in rows:
      r["score"] = r.get("score", 0.0) + weight * r[f"n_{key}"]

  by_cat: dict[str, list[dict]] = {}
  for r in rows:
    by_cat.setdefault(r["cat"], []).append(r)
  for group in by_cat.values():
    group.sort(key=lambda r: -r["score"])

  # one clip per category first, in the order the categories are declared
  chosen: list[dict] = []
  for label, _ in CATEGORIES:
    if len(chosen) >= args.count:
      break
    if by_cat.get(label):
      chosen.append(by_cat[label].pop(0))
  # then allow a second clip from the most dynamic remaining categories
  if len(chosen) < args.count:
    rest = sorted(((label, g[0]) for label, g in by_cat.items() if g), key=lambda kv: -kv[1]["score"])
    for label, r in rest:
      if len(chosen) >= args.count:
        break
      if sum(1 for c in chosen if c["cat"] == label) < args.per_category:
        chosen.append(r)
        by_cat[label].remove(r)

  Path(args.out).write_text("\n".join(f"{r['path']}\t{r['cat']}" for r in chosen) + "\n")
  print(f"{'#':>2}  {'category':10s} {'speed':>6s} {'height':>7s} {'joint':>6s}  clip")
  for i, r in enumerate(chosen, 1):
    name = Path(r["path"]).name.replace(".csv.npz", "").replace(".npz", "")[:58]
    print(f"{i:>2}  {r['cat']:10s} {r['speed']:6.2f} {r['height']:7.2f} {r['joint']:6.1f}  {name}")
  print(f"\nwrote {args.out} ({len(chosen)} clips, {len({c['cat'] for c in chosen})} categories)")


if __name__ == "__main__":
  main()
