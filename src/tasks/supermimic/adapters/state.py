"""Robot-side state providers: the only hand-written part of the task interface.

The 71-dim world-model state and the 33-dim context ``xi`` are the *physical* interface of the world
model. They cannot be derived from a reward configuration and they contain no reward knowledge. Two
implementations:

* :class:`EngineStateProvider` -- the vendored engine's own ``StateIO`` / ``SimContext``. Used whenever
  the environment is the engine's tracking env: it reads the actually-randomised friction, encoder
  bias and torso CoM, which is what the validated runs trained on.
* :class:`MjlabStateProvider` -- a stock ``ManagerBasedRlEnv`` (e.g. the velocity task): read the robot
  data, put the command in ``xi[..., 1:4]`` so the imagined batch can be scored per sampled row.
"""

from __future__ import annotations

from typing import Any

import torch


def _command(env: Any) -> torch.Tensor | None:
  mgr = getattr(env, "command_manager", None)
  if mgr is None:
    return None
  for name in getattr(mgr, "active_terms", []):
    term = mgr.get_term(name)
    cmd = getattr(term, "command", None)
    if torch.is_tensor(cmd) and cmd.ndim == 2:
      return cmd
  return None


def _motion_term(env: Any) -> Any | None:
  mgr = getattr(env, "command_manager", None)
  if mgr is None:
    return None
  for name in getattr(mgr, "active_terms", []):
    term = mgr.get_term(name)
    if hasattr(term, "library") and hasattr(term, "frame"):
      return term
  return None


class EngineStateProvider:
  """Delegates to the engine's ``StateIO`` and ``SimContext`` (exact numbers of the validated runs)."""

  dim: int = 71
  n_joints: int = 29
  xi_dim: int = 33

  def __init__(self, env: Any, foot_geom_regex: str = ".*", anchor_body: str = "torso_link") -> None:
    from supermimic.envs.mdp.state_io import SimContext, StateIO

    self._state_io = StateIO(env)
    self._ctx = SimContext(env, foot_geom_regex, anchor_body)

  def state(self, env: Any) -> torch.Tensor:
    return self._state_io.read()

  def xi(self, env: Any) -> torch.Tensor:
    return self._ctx.read()

  def frame_and_clip(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cmd = env.command_manager.get_term("motion")
    frame = cmd.frame
    return frame, cmd.library.frame_clip[frame]

  def default_joint_pos(self, env: Any) -> torch.Tensor:
    return env.scene["robot"].data.default_joint_pos[0].clone()

  def soft_limits(self, env: Any) -> torch.Tensor:
    return env.scene["robot"].data.soft_joint_pos_limits[0].clone()


class MjlabStateProvider:
  """Stock mjlab env: robot data plus the command carried in ``xi``."""

  dim: int = 71
  n_joints: int = 29
  xi_dim: int = 33

  def state(self, env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat(
      [
        robot.data.root_link_pos_w - env.scene.env_origins,
        robot.data.root_link_quat_w,
        robot.data.joint_pos,
        robot.data.root_link_lin_vel_w,
        robot.data.root_link_ang_vel_b,
        robot.data.joint_vel,
      ],
      dim=-1,
    )

  def xi(self, env: Any) -> torch.Tensor:
    """``xi[..., 0]`` friction scale (1.0: mjlab randomises it in the simulator), ``xi[..., 1:4]`` command."""
    n = int(env.num_envs)
    dev = env.scene["robot"].data.joint_pos.device
    xi = torch.zeros(n, self.xi_dim, device=dev)
    xi[:, 0] = 1.0
    cmd = _command(env)
    if cmd is not None and cmd.shape[-1] >= 3 and cmd.shape[0] == n:
      xi[:, 1:4] = cmd[:, :3]
    return xi

  def frame_and_clip(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(env.num_envs)
    dev = env.scene["robot"].data.joint_pos.device
    z = torch.zeros(n, dtype=torch.long, device=dev)
    motion = _motion_term(env)
    if motion is not None:
      return motion.frame.to(torch.long), motion.library.frame_clip(motion.frame)
    return z, z

  def default_joint_pos(self, env: Any) -> torch.Tensor:
    return env.scene["robot"].data.default_joint_pos[0].clone()

  def soft_limits(self, env: Any) -> torch.Tensor:
    return env.scene["robot"].data.soft_joint_pos_limits[0].clone()


def provider_for(env: Any) -> EngineStateProvider | MjlabStateProvider:
  """The engine's tracking env carries its own state/context readers; everything else uses mjlab data."""
  if bool(getattr(env, "is_engine_env", False)) or hasattr(env, "recorder_manager") and _motion_term(env) is not None:
    try:
      from supermimic.envs.mdp.state_io import SimContext  # noqa: F401

      return EngineStateProvider(env)
    except Exception:  # noqa: BLE001
      pass
  return MjlabStateProvider()
