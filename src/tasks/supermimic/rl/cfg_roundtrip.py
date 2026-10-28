"""Rebuild the nested SuperMimic config from the plain dict mjlab hands to a runner.

``asdict(cfg.agent)`` turns every nested dataclass into a dict, so ``SuperMimicCfg(**flat)``
leaves ``policy``/``env``/``wm_trainer``/... as dicts and the runner fails on the first
attribute access. Rebuild recursively using the dataclass field types.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_type_hints

from src.tasks.supermimic.rl.config import SuperMimicCfg


def _build(cls: type, data: dict[str, Any]) -> Any:
  hints = get_type_hints(cls)
  kwargs: dict[str, Any] = {}
  for f in fields(cls):
    if f.name not in data:
      continue
    value = data[f.name]
    tp = hints.get(f.name)
    if isinstance(value, dict) and tp is not None:
      candidates = [tp] if isinstance(tp, type) else [a for a in get_args(tp) if isinstance(a, type)]
      for cand in candidates:
        if is_dataclass(cand):
          value = _build(cand, value)
          break
    kwargs[f.name] = value
  return cls(**kwargs)


def cfg_from_dict(cfg: dict[str, Any] | SuperMimicCfg) -> SuperMimicCfg:
  if isinstance(cfg, SuperMimicCfg):
    return cfg
  data = cfg.get("supermimic") if isinstance(cfg.get("supermimic"), dict) else cfg
  return _build(SuperMimicCfg, data)


if __name__ == "__main__":
  from dataclasses import asdict

  original = SuperMimicCfg(run_name="x")
  round_trip = cfg_from_dict(asdict(original))
  print("policy type :", type(round_trip.policy).__name__)
  print("env type    :", type(round_trip.env).__name__)
  print("wm type     :", type(round_trip.wm_trainer).__name__)
  print("run_name    :", round_trip.run_name)
