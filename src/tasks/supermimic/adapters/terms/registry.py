"""Analytic projections of mjlab reward terms onto predicted states.

mjlab names a reward term by its function object (``RewardTermCfg.func``). This registry maps that
object to a callable that computes the *same* term on a predicted state. Anything not registered is
either learned (default) or dropped -- never silently mis-implemented.

Every projection returns the **final** value the environment would report for that term, i.e.
``weight * dt * raw``, exactly like ``mjlab.managers.RewardManager`` output. The ``WmSpec`` sums them
and never re-weights, so the imagined reward and the environment's reward are the same number.

Grouping: several terms often share an expensive computation (one FK pass for all body-tracking
kernels). A projection declares ``group``; the group's batched function is evaluated once per step and
memorised on :class:`RewardCtx`; each term then just reads its key.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

_REGISTRY: dict[Any, tuple[Callable[["RewardCtx"], dict[str, torch.Tensor]], str, str]] = {}
_BY_NAME: dict[str, tuple[Callable[["RewardCtx"], dict[str, torch.Tensor]], str, str]] = {}
_FUNCS: dict[str, Any] = {}


def register(func: Any, group: str, family: Callable[["RewardCtx"], dict[str, torch.Tensor]], name: str = "") -> None:
  """Register ``mjlab``'s reward function ``func`` as analytically evaluable.

  family(ctx) must return a dict of *weighted, dt-scaled* per-term values keyed by ``name``.
  """
  _REGISTRY[func] = (family, group, name or getattr(func, "__name__", str(func)))
  _BY_NAME[name or getattr(func, "__name__", str(func))] = _REGISTRY[func]


def register_name(name: str, group: str, family: Callable[["RewardCtx"], dict[str, torch.Tensor]]) -> None:
  """Register an analytic family for a reward *term name* (what ``env.cfg.rewards`` keys by)."""
  entry = (family, group, name)
  _BY_NAME[name] = entry
  func = _FUNCS.get(name)
  if func is not None:
    _REGISTRY[func] = entry


def bind_functions(rewards: dict[str, Any]) -> None:
  """Remember each term's mjlab function object so identity lookups also work."""
  for name, cfg in rewards.items():
    func = getattr(cfg, "func", None)
    if func is not None:
      _FUNCS[name] = func
      entry = _BY_NAME.get(name)
      if entry is not None:
        _REGISTRY[func] = entry


def resolve(func: Any) -> tuple[Callable[["RewardCtx"], dict[str, torch.Tensor]], str, str] | None:
  """Look the implementation up by function object, then by its name.

  The second step matters: ``mjlab``'s re-exported ``mdp`` functions and the vendored engine's own
  reward module define different objects with the same names, so identity alone misclassifies them.
  """
  hit = _REGISTRY.get(func)
  if hit is not None:
    return hit
  hit = _BY_NAME.get(getattr(func, "__name__", ""))
  if hit is not None:
    return hit
  name = getattr(func, "__name__", None)
  if name is None:
    return None
  for key, value in _REGISTRY.items():
    if getattr(key, "__name__", None) == name:
      return value
  return None


def registered_funcs() -> tuple[Any, ...]:
  return tuple(_REGISTRY)


def family_of(ctx, group: str, family: Callable[["RewardCtx"], dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
  """Evaluate a family once per RewardCtx and reuse it for every term of that group."""
  key = f"family::{group}"
  cached = ctx.memo.get(key)
  if cached is None:
    cached = family(ctx)
    ctx.memo[key] = cached
  return cached


def pick(family: Callable[["RewardCtx"], dict[str, torch.Tensor]], name: str, group: str) -> Callable[["RewardCtx"], torch.Tensor]:
  """Term projection: run the family (memoised) and select ``name``."""

  def term(ctx) -> torch.Tensor:
    values = family_of(ctx, group, family)
    if name not in values:
      raise KeyError(f"term '{name}' missing from family '{group}' (has: {sorted(values)})")
    return values[name]

  return term
