"""The single task-facing contract between a mjlab task and the SuperMimic training loop.

mjlab's convention is that the *task* owns reward and termination (``env_cfg.rewards`` /
``env_cfg.terminations``, evaluated by the managers) while the RL side only consumes
``(obs, reward, done)``. SuperMimic cannot consume them from the simulator -- its imagined rollouts
have no simulator -- so the task has to be projected onto predicted states.

That projection is this module. A :class:`WmSpec` is built *from ``env.cfg`` alone* (see
``adapters/from_env.py``), so there is exactly one source of truth and no second hand-written copy of
a reward term. Each reward term is classified into one of three channels:

``analytic``
    the term is a function of the predicted state and is registered in
    ``adapters/terms/registry.py``; evaluated directly.
``learned``
    the term needs contact forces or other simulator-only quantities. A head ``r_k(s, a, xi)`` is
    trained from the per-term values the environment already logs (``buffer.r_terms``).
``unavailable``
    neither possible; dropped loudly. With ``strict`` the run refuses to start.

The same object provides the observation and goal functions, so collection and imagination cannot
disagree about layout either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import torch

TermKind = Literal["analytic", "learned", "unavailable"]


@runtime_checkable
class StateProvider(Protocol):
  """Physical interface of the world model: simulator state and context. Robot-specific only."""

  dim: int

  def state(self, env: Any) -> torch.Tensor: ...
  def xi(self, env: Any) -> torch.Tensor: ...
  def frame_and_clip(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]: ...
  def default_joint_pos(self, env: Any) -> torch.Tensor: ...
  def soft_limits(self, env: Any) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class WmTermSpec:
  """One reward term of the task, projected onto predicted states."""

  name: str
  weight: float
  kind: TermKind
  fn: Callable[["RewardCtx"], torch.Tensor] | None = None
  group: str | None = None
  params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardCtx:
  """Everything an analytic term may need; built once per imagined step and shared."""

  state: torch.Tensor                     # s_{t+1}, (n, 71)
  action: torch.Tensor                    # a_t, (n, 29)
  prev_action: torch.Tensor               # a_{t-1}, (n, 29)
  dt: float
  ref: Any = None                         # ReferenceFrame
  bodies: dict[str, torch.Tensor] | None = None
  command: torch.Tensor | None = None
  default_joint_pos: torch.Tensor | None = None
  soft_limits: torch.Tensor | None = None
  memo: dict[str, Any] = field(default_factory=dict)


class RewardCfgShim:
  """What the engine and the runner read off ``heads.reward_cfg``."""

  def __init__(self, max_step_reward: float, dt: float) -> None:
    self.max_step_reward = float(max_step_reward)
    self.dt = float(dt)


@dataclass(frozen=True, kw_only=True)
class SpecReport:
  """Printed at start-up so a dropped term can never pass silently."""

  analytic: tuple[str, ...]
  learned: tuple[str, ...]
  dropped: tuple[tuple[str, float], ...]
  max_step_reward: float

  def format(self) -> str:
    parts = [
      f"[wm-spec] analytic {len(self.analytic)} | learned {len(self.learned)} | dropped {len(self.dropped)}"
      f" | max_step_reward {self.max_step_reward:.4f}"
    ]
    if self.learned:
      parts.append("[wm-spec] learned (supervised by the env's per-term values): " + ", ".join(self.learned))
    if self.dropped:
      parts.append(
        "[wm-spec] DROPPED (not usable in imagination): "
        + ", ".join(f"{n}({w:+.2f})" for n, w in self.dropped)
      )
    return "\n".join(parts)
