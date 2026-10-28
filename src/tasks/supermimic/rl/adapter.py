"""The one interface a task has to implement to be trainable with SuperMimic.

Two groups of methods:

1. **World-model side** (what the imagined rollout needs, no environment involved):

   ``build_lib()``    reference library -- ``ref(frame)``, ``bodies(frame)``,
                      ``goal_window(frame, k)``, ``last_frame_of(frame)``,
                      ``default_joint_pos``; a mjlab ``MotionCommand`` wrapped by
                      :class:`~src.tasks.supermimic.adapters.tracking.MotionTrackingAdapter`
                      satisfies this.
   ``build_heads()``  analytic reward / termination / forward kinematics evaluated on
                      *predicted* states (differentiable), plus ``reward_cfg`` with
                      ``max_step_reward`` for normalisation.

2. **Environment side** (what the collection step needs):

   ``observation(env)``   ``{"policy": (N, obs_dim), "goal": (N, goal_dim)}``
   ``env_action(a)``      policy output -> environment action (default: identity)
   ``policy_action(a)``   environment action -> policy space (default: identity)
   ``buffer(env)``        replay storage; defaults to :class:`ReplayBuffer`
   ``xi(env)``            sim context ``(N, 33)``; constant when there is no
                          domain randomisation
   ``on_reset(env, ids)`` optional per-reset bookkeeping (observation history, ...)

The adapter may also expose ``goal_dim`` / ``obs_dim`` so the runner can size the
policy; when they are not set they are read from the first observation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class TaskAdapter(Protocol):
  """Task-specific glue between a mjlab environment and the SuperMimic algorithm."""

  #: number of joints of the robot living in the 71-dim ``qpos | qvel`` state
  n_joints: int

  #: ``True`` when the runner should write transitions into the replay buffer itself.
  #: Set it to ``False`` (or omit ``buffer``) when the environment already records them
  #: -- two writers into one ring buffer scramble the sampled windows.
  writes_buffer: bool

  def build_lib(self) -> Any:
    """Reference library (see module docstring)."""
    ...

  def build_heads(self, env: Any) -> Any:
    """Analytic heads: ``reward``, ``terminated``, ``robot_bodies``, ``reward_cfg``."""
    ...

  def observation(self, env: Any) -> dict[str, torch.Tensor]:
    """Current policy input: ``{"policy": (N, obs), "goal": (N, goal)}``."""
    ...

  # ------------------------------------------------------------------ optional
  def env_action(self, action: torch.Tensor) -> torch.Tensor:
    """Map the policy output to the environment action space (default: identity)."""
    return action

  def policy_action(self, action: torch.Tensor) -> torch.Tensor:
    """Map an environment action back to the policy space (default: identity)."""
    return action

  def buffer(self, env: Any) -> Any:
    """Replay buffer; ``None`` lets the runner build the default one."""
    return None

  def xi(self, env: Any) -> torch.Tensor:
    """Sim context ``(N, 33)``; constant nominal values when DR is off."""
    n = int(getattr(env, "num_envs", 1))
    xi = torch.zeros(n, 33, device=getattr(env, "device", "cuda:0"))
    xi[:, 0] = 1.0
    return xi

  def on_reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
    """Per-reset bookkeeping (observation history, action history, ...)."""
    return None
