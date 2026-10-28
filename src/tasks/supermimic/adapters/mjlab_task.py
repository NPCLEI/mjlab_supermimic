"""Bridge for an arbitrary mjlab ``ManagerBasedRlEnv`` with a motion command.

Anything built on mjlab's ``MotionCommand`` (the standard tracking task, the
``mixed_cmd`` task, ...) exposes the same three things SuperMimic needs:

* a reference trajectory: ``command.reference_frames`` / ``command.body_pos_w`` /
  ``command.joint_pos`` -- here converted into the engine's ``MotionLibrary`` API;
* a *goal* for the policy: either the future reference frames (paper default) or the
  command's masked tokens;
* analytic reward and termination criteria evaluated on a *predicted* state.

This adapter implements the frame-based (paper) variant and leaves two clearly marked
hooks -- ``goal_from_state`` and ``reward_terms`` -- for tasks whose goal depends on the
robot state (e.g. anchor-relative tokens). Override them and nothing else changes.

.. code-block:: python

    from src.tasks.supermimic import SuperMimicCfg, SuperMimicRunner
    from src.tasks.supermimic.adapters.mjlab_task import MjlabTrackingAdapter

    env = load_mjlab_env("Mjlab-...-MixedCmd-...")       # any mjlab task with MotionCommand
    adapter = MjlabTrackingAdapter(env)
    runner = SuperMimicRunner(env, SuperMimicCfg(), "logs/x", "cuda:0", adapter=adapter)
    runner.learn()
"""

from __future__ import annotations

from typing import Any

import torch

from supermimic.world_model.heads import ReferenceFrame


class MjlabTrackingAdapter:
  """Adapt a mjlab ``MotionCommand`` task to the SuperMimic algorithm.

  Required command surface (already true for ``MotionCommand``): ``frame``,
  ``time_steps``, ``reference_frames(frames)``, ``joint_pos``/``body_pos_w`` per frame,
  ``num_frames``, ``num_clips``, ``clip_start``.
  """

  n_joints: int = 29

  def __init__(self, env: Any, command_name: str = "motion", goal_frames: int = 10) -> None:
    self.env = env
    self.command_name = command_name
    self.goal_frames = int(goal_frames)
    self._cmd = env.command_manager.get_term(command_name)
    self._lib = _CommandLibrary(self._cmd, goal_frames=self.goal_frames)

  # ------------------------------------------------------------------ task surface
  def build_lib(self) -> Any:
    return self._lib

  def build_heads(self, env: Any) -> Any:
    from .tracking import build_g1_heads

    return build_g1_heads(env)

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

  def observation(self, env: Any) -> dict[str, torch.Tensor]:
    obs = env.observation_manager.compute()
    if isinstance(obs, dict) and "policy" in obs and "goal" in obs:
      return {"policy": obs["policy"], "goal": obs["goal"]}
    raise RuntimeError(
      "the task's observation manager must expose 'policy' and 'goal' groups; "
      "for token goals override MjlabTrackingAdapter.goal_from_state"
    )

  def frame_and_clip(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    frame = self._cmd.time_steps
    return frame.long(), torch.zeros_like(frame, dtype=torch.long)

  def on_reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
    return None

  # ------------------------------------------------------------------ extension hooks
  def goal_from_state(self, state: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Goal vector for the policy. Default: the paper's future-frame features."""
    return self._lib.goal_window(frames, self.goal_frames)

  def reward_terms(self, states: torch.Tensor, actions: torch.Tensor, prev_actions: torch.Tensor, frames: torch.Tensor):
    """Analytic reward terms on predicted states (see ``heads.MjlabAnalyticHeads``)."""
    return self.build_heads(self.env).reward(states, actions, prev_actions, self._lib.ref(frames))


class _CommandLibrary:
  """``MotionLibrary``-compatible view over a mjlab ``MotionCommand``."""

  def __init__(self, command: Any, goal_frames: int = 10) -> None:
    self._cmd = command
    self.goal_frames = int(goal_frames)
    self.default_joint_pos = self._default_joint_pos()
    self.body_names = tuple(getattr(command, "body_names", ()))

  def _default_joint_pos(self) -> torch.Tensor:
    # the command exposes the reference joint positions per frame; frame 0 of every clip
    # is a valid "default" for the goal's joint offset (the engine only needs a constant)
    try:
      frames = torch.zeros(1, dtype=torch.long, device=self._cmd.device)
      return self._cmd.reference_frames(frames).joint_pos[0].detach()
    except Exception:  # noqa: BLE001
      return torch.zeros(29, device=getattr(self._cmd, "device", "cuda:0"))

  # ------------------------------------------------------------------ engine API
  @property
  def clips(self):
    return getattr(self._cmd, "clips", [])

  def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
    return self._cmd.num_frames(frame) if hasattr(self._cmd, "num_frames") else frame

  def ref(self, frame: torch.Tensor) -> ReferenceFrame:
    r = self._cmd.reference_frames(frame)
    return ReferenceFrame(None, None, None, None, r.body_pos_w[..., 0, :], r.body_quat_w[..., 0, :])

  def bodies(self, frame: torch.Tensor):
    return self._cmd.reference_frames(frame)

  def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
    """``(N, k * 68)`` flattened future-frame features (paper Table A.1 goal)."""
    out = []
    for i in range(k):
      r = self._cmd.reference_frames(torch.clamp(frame + i, max=self._cmd.num_frames(frame) - 1))
      out.append(r.joint_pos)
    return torch.cat(out, dim=-1)
