#!/usr/bin/env python
"""Train a mjlab motion-tracking task with SuperMimic.

  python scripts/train_tracking.py --adapter tracking --data.npz-dir motion_lib_300 \
      --env.num-envs 2048 --env.anchor-obs --schedule.max-hours 8

Equivalent to:  PYTHONPATH=. python -m src.tasks.supermimic.train <args>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks.supermimic.train import main  # noqa: E402
from src.tasks.supermimic.rl.config import SuperMimicCfg  # noqa: E402

if __name__ == "__main__":
  import tyro

  main(tyro.cli(SuperMimicCfg))
