"""Registration of the analytic families: motion tracking (8 terms) and command tracking (6 terms).

Called once when a spec is built from an env config; the registry is keyed by the mjlab reward
*functions*, so a task that uses these functions is automatically analytic -- no per-task code.
"""

from __future__ import annotations

from typing import Any

import torch

from src.tasks.supermimic.adapters.terms.registry import register_name


def register_tracking(heads: Any) -> None:
  """Register the body-tracking reward terms against the engine's analytic implementation.

  ``heads`` is the engine's ``AnalyticHeads`` subclass that has already read weights and stds back from
  ``env.cfg``; the family therefore reproduces the environment's numbers exactly. Registration is by
  *term name* because the functions live in the engine's own reward module, not in ``mjlab.envs.mdp``.
  """

  def family(ctx) -> dict[str, torch.Tensor]:
    bodies = ctx.bodies if ctx.bodies is not None else heads.robot_bodies(ctx.state)
    return heads.reward(ctx.state, ctx.action, ctx.prev_action, ctx.ref, bodies)

  for name in (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
  ):
    register_name(name, "tracking", family)


def register_command(heads: Any) -> None:
  """Register the velocity-tracking family (pure state functions, no FK)."""

  def family(ctx) -> dict[str, torch.Tensor]:
    return heads.reward(ctx.state, ctx.action, ctx.prev_action, ctx.ref, ctx.bodies, xi=ctx.command)

  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "body_ang_vel",
    "dof_pos_limits",
    "action_rate_l2",
  ):
    register_name(name, "command", family)
