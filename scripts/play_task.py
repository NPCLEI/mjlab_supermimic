#!/usr/bin/env python
"""Play a trained policy in mjlab's native viser web viewer.

  python scripts/play_task.py --ckpt logs/run/ckpt_latest.pt --npz-dir motion_lib_1 --port 8097
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/tasks/supermimic/engine"))

import runpy  # noqa: E402

runpy.run_path(str(Path(__file__).with_name("play.py")), run_name="__main__")
