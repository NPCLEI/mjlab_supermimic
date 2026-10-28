"""Reference adapter: G1 motion tracking on any npz motion library.

This is the setting that reproduces the paper's headline result (one policy, hundreds of
motions, ~3 h). The environment, reference library and analytic heads are the vendored
reference implementation's own -- untouched -- so this adapter is a thin, well-tested
mapping. Use it as the template for other tasks.

.. code-block:: python

    cfg = SuperMimicCfg(...)
    env = MotionTrackingAdapter.build_env(cfg, npz_dir="motion_lib_300")
    adapter = MotionTrackingAdapter(env)
    runner = SuperMimicRunner(env, cfg, "logs/run", "cuda:0", adapter=adapter)
    runner.learn()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import XI_DIM, SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads

FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"


def build_g1_heads(env: Any) -> AnalyticHeads:
  """Analytic reward / termination / FK heads for the G1 tracked bodies.

  ``EnvSyncedHeads`` reads the reward weights, reward stds, dt and the termination
  thresholds back from ``env.cfg``: tuning the task's reward / termination does not
  require touching this file, let alone the framework.
  """
  from .heads import EnvSyncedHeads

  robot = env.scene["robot"]
  fk = G1Kinematics(load_g1_model(), device=env.device)
  return EnvSyncedHeads(
    env,
    fk,
    TRACKED_BODIES,
    ANCHOR_BODY,
    ROOT_BODY,
    robot.data.soft_joint_pos_limits[0].clone(),
    robot.data.default_joint_pos[0].clone(),
  )


class MotionTrackingAdapter:
  """Task adapter for ``supermimic.envs`` motion tracking."""

  n_joints: int = 29
  #: the engine's environment already records transitions into its own replay buffer via
  #: the recorder manager, so the runner must not write rows itself (that would interleave
  #: two writers into the same ring and scramble the windows).
  writes_buffer: bool = False
  #: the engine's env keeps mjlab's ``auto_reset=True``: ``env.step`` already resets
  #: finished episodes (and the recorder hooks write the terminal rows), so the
  #: runner must not reset them again.
  auto_reset: bool = True

  def __init__(self, env: Any) -> None:
    self.env = env
    self._state_io = StateIO(env)
    self._ctx = SimContext(env, FOOT_GEOM_REGEX, "torso_link")
    self._lib = None
    self._heads = None

  # ------------------------------------------------------------------ construction
  @staticmethod
  def build_env(
    cfg: Any,
    npz_dir: str | Path | None = None,
    clips: list[str] | None = None,
    compact_library: Path | None = None,
    device: str | None = None,
  ) -> Any:
    """Build the tracking environment from a motion library (npz dir or packed .pt)."""
    if compact_library is not None:
      files = [compact_library]
      compact = True
    else:
      directory = Path(npz_dir or cfg.data.npz_dir)
      files = discover_npz(directory, limit=cfg.data.limit, names=clips or cfg.data.clips, manifest=cfg.data.manifest)
      compact = False
    if not files:
      raise FileNotFoundError(f"no clips under {npz_dir or cfg.data.npz_dir}")
    env = make_env(
      SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files],
        num_envs=cfg.env.num_envs,
        future_frames=cfg.env.future_frames,
        episode_length_s=cfg.env.episode_length_s,
        action_noise_std=cfg.env.action_noise_std,
        obs_noise=cfg.env.obs_noise,
        reset_noise=cfg.env.reset_noise,
        domain_randomization=cfg.env.domain_randomization,
        sampling_mode=cfg.env.sampling_mode,
        replay_capacity=cfg.env.replay_capacity,
        compact_library=compact,
        anchor_obs=cfg.env.anchor_obs,
        seed=cfg.seed,
      ),
      device=device or cfg.device,
    )
    return env

  # ------------------------------------------------------------------ task surface
  def build_lib(self) -> Any:
    if self._lib is None:
      self._lib = self.env.command_manager.get_term("motion").library
    return self._lib

  def build_heads(self, env: Any) -> Any:
    if self._heads is None:
      self._heads = build_g1_heads(env)
    return self._heads

  def observation(self, env: Any) -> dict[str, torch.Tensor]:
    obs = env.observation_manager.compute() if hasattr(env, "observation_manager") else None
    if obs is None or not isinstance(obs, dict):
      raise RuntimeError("the tracking env must return {'policy', 'goal'} observations")
    return {"policy": obs["policy"], "goal": obs["goal"]}

  def state(self, env: Any) -> torch.Tensor:
    return self._state_io.read()

  def xi(self, env: Any) -> torch.Tensor:
    return self._ctx.read()

  def frame_and_clip(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cmd = env.command_manager.get_term("motion")
    frame = cmd.frame
    return frame, cmd.library.frame_clip[frame]

  def on_reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
    # the reference env keeps its own observation/action history
    return None

  def buffer(self, env: Any) -> Any:
    return env.recorder_manager.get_term("replay").buffer
