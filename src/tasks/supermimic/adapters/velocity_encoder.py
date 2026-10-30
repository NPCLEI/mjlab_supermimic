"""Velocity controller on a pretrained motion decoder -- the paper's Appendix-B path.

Why this exists
---------------
Every from-scratch attempt to learn ``Mjlab-Velocity-Flat`` with SuperMimic stalled at "stand still,
5 s episodes, 6-10% of commands tracked": with no gait prior the world model's action sensitivity
(measured: 11.6% of reality) leaves stepping worthless in imagination, while standing already earns the
upright term in full.

The paper's Sec. 5.5 / Appendix B velocity controller does something different and much easier: it keeps
a *motion-tracking* decoder (trained on reference motions, so it already knows how to walk) and learns a
new encoder ``z = enc_v(proprio_history_930, command_3)``. One reproduction reports 652/672 commanded
conditions with that recipe, versus 650/672 for the paper.

Layout compatibility
--------------------
The tracking decoder consumes ``930 = 10 x 93`` proprioception in term-major order with the terms
``[q - q0 (29), qd (29), omega_body (3), gravity_body (3), prev_action (29)]``. The velocity env's own
actor group interleaves the command and uses a different term order, so this module rebuilds the exact
engine layout from the velocity environment's state, which lets the pretrained decoder be reused as-is.
"""

from __future__ import annotations

import torch

OBS_TERM_DIMS = (29, 29, 3, 3, 29)
OBS_FRAME = sum(OBS_TERM_DIMS)  # 93
HISTORY = 10
OBS_DIM = OBS_FRAME * HISTORY  # 930
COMMAND_SCALE = (2.0, 1.32, 3.35)


# ---------------------------------------------------------------------- observation
def observation_frame(q, qd, omega_body, gravity_body, prev_action, default_joint_pos) -> torch.Tensor:
  """The engine's 93-dim frame, in the same term order the tracking decoder was trained on."""
  return torch.cat([q - default_joint_pos, qd, omega_body, gravity_body, prev_action], dim=-1)


def flatten_term_major(hist: torch.Tensor) -> torch.Tensor:
  """(H, N, 93) oldest -> newest  to  (N, 930) term-major, matching ``flatten_history_term_major``."""
  h, n, _ = hist.shape
  chunks, start = [], 0
  for d in OBS_TERM_DIMS:
    chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(n, h * d))
    start += d
  return torch.cat(chunks, dim=-1)


class HistoryBuffer:
  """Ring buffer of the last ``HISTORY`` 93-dim frames per environment, newest last."""

  def __init__(self, n: int, device) -> None:
    self.hist = torch.zeros(HISTORY, n, OBS_FRAME, device=device)
    self.n = n

  def reset(self, ids: torch.Tensor | None = None) -> None:
    if ids is None:
      self.hist.zero_()
    else:
      self.hist[:, ids] = 0.0

  @torch.no_grad()
  def push(self, frame: torch.Tensor) -> None:
    self.hist = torch.roll(self.hist, shifts=-1, dims=0)
    self.hist[-1] = frame

  def flat(self) -> torch.Tensor:
    return flatten_term_major(self.hist)


# ---------------------------------------------------------------------- rewards (paper Table A.6 style)
def velocity_reward_terms(
  *,
  planar_err: torch.Tensor,
  yaw_err: torch.Tensor,
  action: torch.Tensor,
  prev_action: torch.Tensor,
  joint_limit_excess: torch.Tensor,
  sideways_lean: torch.Tensor,
  omega_heading_xy: torch.Tensor,
  dt: float,
  cfg,
) -> dict[str, torch.Tensor]:
  """Rewards shaped so walking is never punished for leaning forward.

  ``upright``-style terms (reward for being vertical) are deliberately absent: a fully upright torso is
  what standing looks like, so such a term pays the standing local optimum in full. Instead a penalty on
  *sideways* lean keeps the robot from falling over while permitting the forward lean that locomotion
  needs.
  """
  terms = {
    "planar_velocity": cfg.w_planar * torch.exp(-planar_err.pow(2).sum(-1) / (cfg.sigma_v ** 2)),
    "yaw_rate": cfg.w_yaw * torch.exp(-yaw_err.pow(2) / (cfg.sigma_w ** 2)),
    "action_smooth": -cfg.w_action_smooth * (action - prev_action).pow(2).sum(-1),
    "joint_limit": -cfg.w_joint_limit * joint_limit_excess.sum(-1),
    "torso_lean": -cfg.w_lean * sideways_lean.pow(2),
    "torso_yaw_rate": -cfg.w_omega * omega_heading_xy.pow(2).sum(-1),
  }
  out = {k: v * dt for k, v in terms.items()}
  out["total"] = sum(out.values())
  return out


def sideways_lean(root_quat: torch.Tensor) -> torch.Tensor:
  """Component of gravity along the robot's lateral axis: 0 upright, +-1 lying on a side.

  Forward lean (rotation about the lateral axis) leaves this at zero, which is exactly the point.
  """
  g_world = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device)
  g_world = g_world.expand(root_quat.shape[:-1] + (3,))
  # rotate gravity into the body frame using the conjugate quaternion
  x, y, z, w = -root_quat[..., 0], -root_quat[..., 1], -root_quat[..., 2], root_quat[..., 3]
  q_vec = torch.stack([x, y, z], dim=-1)
  t = 2.0 * torch.cross(q_vec, g_world, dim=-1)
  g_body = g_world + w[..., None] * t + torch.cross(q_vec, t, dim=-1)
  return g_body[..., 1]  # lateral axis
