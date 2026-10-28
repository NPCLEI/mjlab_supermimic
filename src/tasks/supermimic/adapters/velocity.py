"""Velocity-tracking adapter: the framework is not tied to motion tracking.

The stock mjlab velocity task (`Mjlab-Velocity-Flat-Unitree-G1`) differs from a tracking task
in every task-specific dimension:

* goal: a **4-dim velocity command** resampled every 3-8 s, not a window of future frames;
* rewards: velocity / posture terms (14 of them), not body-tracking kernels;
* termination: ``fell_over`` (tilt > 70 deg), not ``root_height``.

SuperMimic only needs four things from a task -- a 71-dim state, a goal vector, an *analytic*
reward and an *analytic* termination on predicted states -- plus the analytic observation the
policy consumes. This adapter supplies exactly those; the runner, world model and imagination
are untouched.

Terms that cannot be scored in imagination (they need contact forces, foot forces or air
time) are excluded and listed in :data:`NON_ANALYTIC`. That is a property of the method, not
of this framework: the imagined reward must be a differentiable function of the state.
"""

from __future__ import annotations

from typing import Any

import torch

from supermimic.world_model.heads import ReferenceFrame

NON_ANALYTIC = (
  "air_time", "foot_clearance", "foot_slip", "soft_landing", "self_collisions", "angular_momentum",
)
ANALYTIC = (
  "track_linear_velocity", "track_angular_velocity", "upright",
  "body_ang_vel", "dof_pos_limits", "action_rate_l2",
)
#: "pose" is deliberately absent: mjlab compares against a commanded reference pose, so the
#: default-stance proxy would reward standing still and fight the velocity objective.
POSE_IS_PROXY = True
TILT_LIMIT = 1.2217304763960306  # mjlab's fell_over limit (70 deg)
N_JOINTS = 29


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply_inverse

  # the world model predicts the quaternion as four free numbers: |q| drifts in imagination and an
  # unnormalised quaternion lets the policy game `upright` / the tilt termination
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  g = torch.tensor([0.0, 0.0, -1.0], dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 3)
  return quat_apply_inverse(q, g)


def _tilt(q: torch.Tensor) -> torch.Tensor:
  g = projected_gravity(q)
  return torch.acos((-g[..., 2]).clamp(-1.0, 1.0))


class VelocityHeads:
  """Analytic reward / termination / observation for the velocity task."""

  anchor_index = 0
  reward_term_names = ANALYTIC
  action_bound = 1.0  # the env clips joint_pos actions to [-1, 1]


  def __init__(self, env: Any, actor_dim: int, command_name: str = "twist") -> None:
    self.env = env
    self.command_name = command_name
    # the environment's actor group does not know the base velocity; the adapter appends the error
    self.actor_dim = int(actor_dim) + self.anchor_dim
    robot = env.scene["robot"]
    self.default_joint_pos = robot.data.default_joint_pos[0].clone()
    self.soft_limits = robot.data.soft_joint_pos_limits[0].clone()
    self.dt = float(env.step_dt)
    rewards = getattr(env.cfg, "rewards", {}) or {}
    self.weights = {k: float(getattr(rewards.get(k), "weight", 0.0)) for k in ANALYTIC}
    self.stds = {}
    for k in ANALYTIC:
      params = getattr(rewards.get(k), "params", None) or {}
      self.stds[k] = float(params.get("std", 0.5))
    positives = sum(w for w in self.weights.values() if w > 0.0)
    self.reward_cfg = type("RewardCfg", (), {"max_step_reward": self.dt * positives})()

  # ------------------------------------------------------------------ helpers
  @staticmethod
  def _split(s: torch.Tensor):
    return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

  def _command(self, xi: torch.Tensor | None = None) -> torch.Tensor:
    if xi is not None and xi.shape[-1] >= 4:
      return xi[..., 1:4]
    return self.env.command_manager.get_term(self.command_name).command

  #: extra anchor-like features appended to the actor observation (velocity error + speed)
  anchor_dim: int = 4

  def anchor_feature(self, s_next: torch.Tensor, xi: torch.Tensor | None = None) -> torch.Tensor:
    """How far the predicted state is from the commanded motion (tracking's anchor analogue)."""
    _, _, _, v, w, _ = self._split(s_next)
    cmd = self._command(xi).to(s_next.dtype)
    err_v = cmd[..., :2] - v[..., :2]
    err_w = (cmd[..., 2] - w[..., 2]).unsqueeze(-1)
    speed = cmd[..., :2].norm(dim=-1, keepdim=True)
    return torch.cat([err_v, err_w, speed], dim=-1)

  # ------------------------------------------------------------------ analytic observation
  def observation(self, s_next: torch.Tensor, prev_action: torch.Tensor, xi, default_joint_pos) -> torch.Tensor:
    """The velocity task's actor observation, computed from a predicted state.

    Layout matches the env's ``actor`` group: base_ang_vel(3) | projected_gravity(3) |
    command(4) | joint_pos-default(29) | joint_vel(29) | actions(29).
    """
    _, q, qj, _, w, qd = self._split(s_next)
    obs = torch.cat([w, projected_gravity(q), (self._command().to(s_next.dtype).expand(s_next.shape[0], -1)
                        if xi is None else self._command(xi).to(s_next.dtype)),
                       qj - self.default_joint_pos, qd, prev_action,
                       self.anchor_feature(s_next, xi)], dim=-1)
    if obs.shape[-1] != self.actor_dim:
      raise RuntimeError(f"analytic observation {obs.shape[-1]} != actor group {self.actor_dim}")
    return obs

  def goal(self, start, ref, anchor_pose, future_frames: int) -> torch.Tensor:
    """The imagined batch's goal: each sampled row's own velocity command, repeated."""
    cmd = start.xi[..., 1:4]
    return cmd.repeat(1, int(future_frames))

  # ------------------------------------------------------------------ heads API
  @staticmethod
  def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref=None) -> torch.Tensor:
    """Smooth tilt proxy ``1 + g_z`` (0 upright, 2 upside down).

    ``acos`` must not appear here: at ``-g_z == 1`` its derivative is infinite, which turns the
    world-model gradients into NaN even though the loss itself is finite.
    """
    g = projected_gravity(root_quat)
    tilt_proxy = 1.0 + g[..., 2].clamp(-1.0, 1.0)
    return torch.stack([torch.zeros_like(tilt_proxy), tilt_proxy], dim=-1)

  def terminated(self, s: torch.Tensor, ref=None, height_thr: float | None = None, tilt_thr: float | None = None):
    _, q, *_ = self._split(s)
    return _tilt(q) > (TILT_LIMIT if tilt_thr is None else tilt_thr)

  @staticmethod
  def root_error_features(s: torch.Tensor, ref=None) -> torch.Tensor:
    return torch.zeros(*s.shape[:-1], 4, device=s.device, dtype=s.dtype)

  @staticmethod
  def robot_bodies(s: torch.Tensor) -> dict[str, torch.Tensor]:
    p, q, _, v, w, _ = VelocityHeads._split(s)
    return {
      "pos": p[:, None, :],
      "quat": q[:, None, :],
      "lin_vel": v[:, None, :],
      "ang_vel": w[:, None, :],
    }

  def reward(self, s_next, action, prev_action, ref, bodies=None, xi=None) -> dict[str, torch.Tensor]:
    _, q, qj, v, w, qd = self._split(s_next)
    g = projected_gravity(q)
    cmd = self._command(xi).to(s_next.dtype)
    dt = self.dt
    terms = {}
    terms["track_linear_velocity"] = dt * self.weights["track_linear_velocity"] * torch.exp(
      -((v[..., :2] - cmd[..., :2]) ** 2).sum(-1) / self.stds["track_linear_velocity"] ** 2)
    terms["track_angular_velocity"] = dt * self.weights["track_angular_velocity"] * torch.exp(
      -((w[..., 2] - cmd[..., 2]) ** 2) / self.stds["track_angular_velocity"] ** 2)
    terms["upright"] = dt * self.weights["upright"] * torch.exp(
      -(g[..., :2] ** 2).sum(-1) / self.stds["upright"] ** 2)
    terms["body_ang_vel"] = dt * self.weights["body_ang_vel"] * (w ** 2).sum(-1)
    lo, hi = self.soft_limits[..., 0], self.soft_limits[..., 1]
    # the predicted joint angle is unbounded; cap the violation so one bad prediction cannot
    # dominate the whole imagined return
    violation = ((lo - qj).clamp_min(0.0) + (qj - hi).clamp_min(0.0)).clamp_max(1.0)
    terms["dof_pos_limits"] = dt * self.weights["dof_pos_limits"] * violation.sum(-1)
    terms["action_rate_l2"] = dt * self.weights["action_rate_l2"] * ((action - prev_action) ** 2).sum(-1)
    total = terms["track_linear_velocity"]
    for k, val in terms.items():
      if k != "track_linear_velocity":
        total = total + val
    terms["total"] = total
    return terms


class VelocityLib:
  """Engine-facing ``lib``: the goal is the velocity command, references are placeholders."""

  def __init__(self, env: Any, command_name: str = "twist", n_bodies: int = 1) -> None:
    self.env = env
    self.command_name = command_name
    self.n_bodies = n_bodies
    self.device = env.scene["robot"].data.joint_pos.device
    self.default_joint_pos = env.scene["robot"].data.default_joint_pos[0].clone()

  def _cmd(self) -> torch.Tensor:
    return self.env.command_manager.get_term(self.command_name).command

  def goal_window(self, frame, k: int) -> torch.Tensor:
    return self._cmd().repeat(1, k)

  def _zeros(self, m: int):
    pos = torch.zeros(m, self.n_bodies, 3, device=self.device)
    quat = torch.zeros(m, self.n_bodies, 4, device=self.device)
    quat[..., 0] = 1.0
    root_pos = torch.zeros(m, 3, device=self.device)
    root_quat = torch.zeros(m, 4, device=self.device)
    root_quat[..., 0] = 1.0
    return pos, quat, root_pos, root_quat

  def ref(self, frame) -> ReferenceFrame:
    m = int(frame.numel())
    pos, quat, rp, rq = self._zeros(m)
    return ReferenceFrame(pos, quat, pos, pos, rp, rq)

  def bodies(self, frame):
    m = int(frame.numel())
    pos, quat, _, _ = self._zeros(m)

    class _B:
      pass

    b = _B()
    b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w = pos, quat, pos, pos
    return b

  def last_frame_of(self, frame):
    return frame

  def frame_clip(self, frame):
    return torch.zeros_like(frame, dtype=torch.long)


class VelocityAdapter:
  """TaskAdapter for the stock mjlab velocity task."""

  n_joints: int = N_JOINTS
  #: observation history: the actor group has none, but the policy needs one to know its own speed
  history: int = 1
  writes_buffer: bool = True     # the stock task has no replay recorder
  auto_reset: bool = True        # mjlab resets finished episodes; termination is marked inline

  def __init__(self, env: Any, command_name: str = "twist") -> None:
    self.env = env
    self.command_name = command_name
    self._lib = VelocityLib(env, command_name)
    actor_dim = int(env.observation_manager.group_obs_dim["actor"][0])
    self._heads = VelocityHeads(env, actor_dim, command_name)
    self._hist: list[torch.Tensor] = []

  # ------------------------------------------------------------------ construction
  @staticmethod
  def build_env(cfg: Any, task_id: str | None = None) -> Any:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(task_id or cfg.task_id, play=False)
    env_cfg.scene.num_envs = int(cfg.env.num_envs)
    env_cfg.seed = int(cfg.seed)
    return ManagerBasedRlEnv(env_cfg, device=cfg.device, render_mode=None)

  # ------------------------------------------------------------------ task surface
  def build_lib(self):
    return self._lib

  def build_heads(self, env):
    return self._heads

  def state(self, env) -> torch.Tensor:
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

  def observation(self, env) -> dict[str, torch.Tensor]:
    obs = env.observation_manager.compute()
    actor = obs["actor"]
    # tracking appends anchor-relative quantities so the policy can correct drift; the velocity
    # analogue is the live velocity error, which the actor group does not contain
    actor = torch.cat([actor, self._heads.anchor_feature(self.state(env), self.xi(env))], dim=-1)
    # frame-major history, matching what ``imagine`` builds from the sampled history states
    self._hist.append(actor.detach())
    if len(self._hist) > self.history:
      self._hist.pop(0)
    while len(self._hist) < self.history:
      self._hist.insert(0, self._hist[0])
    actor = torch.cat(self._hist, dim=-1)
    # the goal is the current velocity command, repeated over the imagination horizon
    frames = int(getattr(self.env.cfg, "future_frames", 10) or 10)
    goal = self._lib.goal_window(torch.zeros(actor.shape[0], dtype=torch.long, device=actor.device), frames)
    return {"policy": actor, "goal": goal}

  def frame_and_clip(self, env):
    n = int(env.num_envs)
    z = torch.zeros(n, dtype=torch.long, device=self._lib.device)
    return z, z

  def xi(self, env):
    """xi[:, 0] = friction scale, xi[:, 1:4] = the velocity command (per row)."""
    n = int(env.num_envs)
    xi = torch.zeros(n, 33, device=self._lib.device)
    xi[:, 0] = 1.0
    cmd = self._lib._cmd()
    if cmd.shape[-1] >= 3 and n == cmd.shape[0]:
      xi[:, 1:4] = cmd[:, :3]
    return xi

  def on_reset(self, env, env_ids=None) -> None:
    self._hist.clear()
