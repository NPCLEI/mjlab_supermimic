"""``SuperMimicRunner`` -- the PPO-like entry point.

Same contract as mjlab's on-policy runner: build it around an environment plus a config,
call :meth:`learn`, then :meth:`save`. The algorithm (world model, imagined rollout,
objective, GAE critic, UTD schedule) is the vendored reference implementation, unchanged;
everything task specific comes from the :class:`TaskAdapter`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.policy.nets import Critic, Policy
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel

STATE_DIM = 71
XI_DIM = 33


class SuperMimicRunner:
  """Train any mjlab task with SuperMimic.

  Args:
    env: a ``ManagerBasedRlEnv`` (must allow manual resets: ``auto_reset = False``).
    cfg: a :class:`TrainConfig` / :class:`~src.tasks.supermimic.rl.config.SuperMimicCfg`.
    log_dir: run directory (checkpoints, tensorboard, config.json).
    device: torch device string.
    adapter: task adapter (see :class:`~src.tasks.supermimic.rl.adapter.TaskAdapter`).
      When omitted, ``lib`` and ``heads`` must be passed explicitly (engine-native use).
    lib, heads: override the adapter's reference library / analytic heads.
  """

  def __init__(
    self,
    env: Any,
    cfg: TrainConfig,
    log_dir: str | Path,
    device: str = "cuda:0",
    adapter: Any | None = None,
    lib: Any | None = None,
    heads: Any | None = None,
  ) -> None:
    self.cfg = cfg
    self.env = env
    self.adapter = adapter
    self.device = device
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    self.is_main = True
    self.writer = SummaryWriter(str(self.log_dir / "tb"))
    (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    if adapter is None:
      # reflect the task from env.cfg (rewards, terminations, observation, command) -- this is the
      # only place a task is interpreted, and it takes no per-task code
      from src.tasks.supermimic.adapters.from_env import build_wm_spec

      adapter = build_wm_spec(env, cfg)

    # ---- task surface -----------------------------------------------------
    self.lib = lib if lib is not None else adapter.build_lib()
    self.heads = heads if heads is not None else adapter.build_heads(env)
    self.n_joints = int(getattr(adapter, "n_joints", 29))
    act_dim = int(env.action_space.shape[-1]) if hasattr(env, "action_space") else self.n_joints

    obs = adapter.observation(env)
    cfg.policy.goal_dim = int(getattr(adapter, "goal_dim", 0) or obs["goal"].shape[-1])
    cfg.policy.obs_dim = int(obs["policy"].shape[-1])
    cfg.policy.act_dim = act_dim

    # ---- algorithm --------------------------------------------------------
    self.max_step_reward = float(self.heads.reward_cfg.max_step_reward)
    self.reward_opt = None
    if getattr(cfg, "reward_model", False) or getattr(adapter, "reward_from_env", False):
      from src.tasks.supermimic.adapters.learned_reward import (
        LearnedRewardHeads, RewardModel, RewardModelCfg,
      )

      rm_cfg = getattr(cfg, "reward_model_cfg", None) or RewardModelCfg()
      model = RewardModel(71, int(cfg.policy.act_dim or 29), 33, rm_cfg.hidden).to(device)
      self.heads = LearnedRewardHeads(self.heads, model, target_scale=float(self.max_step_reward or 1.0))
      self.reward_opt = torch.optim.Adam(model.parameters(), lr=rm_cfg.lr, weight_decay=rm_cfg.weight_decay)
      self.reward_cfg_rm = rm_cfg
      self._reward_scales_set = False
      print("[INFO] learned reward model enabled (r̂(s,a,xi) supervises the imagined reward)", flush=True)
    self.wm = WorldModel(cfg.world_model, self.heads, device)
    self.policy = Policy(cfg.policy).to(device)
    self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(device)
    self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
    self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)

    self.buffer = self._buffer_hook()
    if self.buffer is None:
      self.buffer = ReplayBuffer(
        int(env.num_envs), int(cfg.env.replay_capacity), STATE_DIM, act_dim, XI_DIM, device
      )

    self._adds_action_noise = bool(getattr(adapter, 'adds_action_noise', False))
    self._auto_reset = bool(getattr(adapter, 'auto_reset', False))
    self._writes_buffer = bool(getattr(adapter, 'writes_buffer', True))
    self.max_step_reward = float(self.heads.reward_cfg.max_step_reward)
    n = int(env.num_envs)
    self._prev_s = torch.zeros(n, STATE_DIM, device=device)
    self._prev_xi = torch.zeros(n, XI_DIM, device=device)
    self._prev_frame = torch.zeros(n, dtype=torch.long, device=device)
    self._prev_clip = torch.zeros(n, dtype=torch.long, device=device)
    self._ep_start = torch.ones(n, dtype=torch.bool, device=device)
    self.ep_reward_sum = torch.zeros(n, device=device)
    self.ep_len = torch.zeros(n, device=device)
    self.iteration = 0
    self.num_updates = 0
    self._utd_acc = 0.0
    self._skipped_updates = 0
    n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    print(
      f"[INFO] SuperMimicRunner | wm {n_params(self.wm) / 1e6:.2f}M "
      f"policy {n_params(self.policy) / 1e6:.2f}M critic {n_params(self.critic) / 1e6:.3f}M "
      f"| obs {cfg.policy.obs_dim} goal {cfg.policy.goal_dim} act {act_dim}",
      flush=True,
    )

    # resume LAST: load() restores iteration / num_updates, which are otherwise
    # overwritten by the defaults set above.
    if cfg.resume is not None:
      self.load(cfg.resume)

  # ------------------------------------------------------------------ task hooks
  def _hook(self, name: str):
    """Task hook with the protocol's default when the adapter omits it."""
    fn = getattr(self.adapter, name, None) if self.adapter is not None else None
    return fn

  def _state(self) -> torch.Tensor:
    fn = self._hook("state")
    if fn is None:
      raise NotImplementedError("the adapter must provide state(env) -> (N, 71)")
    return fn(self.env)

  def _xi(self) -> torch.Tensor:
    fn = self._hook("xi")
    if fn is not None:
      return fn(self.env)
    xi = torch.zeros(int(self.env.num_envs), XI_DIM, device=self.device)
    xi[:, 0] = 1.0
    return xi

  def _frame_clip(self) -> tuple[torch.Tensor, torch.Tensor]:
    fn = self._hook("frame_and_clip")
    if fn is not None:
      return fn(self.env)
    n = int(self.env.num_envs)
    z = torch.zeros(n, dtype=torch.long, device=self.device)
    return z, z

  def _observation(self) -> dict[str, torch.Tensor]:
    return self.adapter.observation(self.env)

  def _env_action(self, a: torch.Tensor) -> torch.Tensor:
    fn = self._hook("env_action")
    return fn(a) if fn is not None else a

  def _buffer_hook(self) -> Any:
    fn = self._hook("buffer")
    return fn(self.env) if fn is not None else None

  def _on_reset(self, ids: torch.Tensor | None) -> None:
    fn = self._hook("on_reset")
    if fn is not None:
      fn(self.env, ids)

  # ------------------------------------------------------------------ checkpoints
  def save(self, name: str = "ckpt_latest.pt") -> Path:
    path = self.log_dir / name
    torch.save(
      {
        "iteration": self.iteration,
        "num_updates": self.num_updates,
        "wm": self.wm.state_dict(),
        "policy": self.policy.state_dict(),
        "critic": self.critic.state_dict(),
        "wm_opt": self.wm_trainer.opt.state_dict(),
        "policy_opt": self.policy_trainer.opt.state_dict(),
        "critic_opt": self.policy_trainer.critic_opt.state_dict(),
        "config": asdict(self.cfg),
      },
      path,
    )
    return path

  def load(self, path: str | Path) -> None:
    ck = torch.load(path, map_location=self.device, weights_only=False)
    self.wm.load_state_dict(ck["wm"])
    self.policy.load_state_dict(ck["policy"])
    self.critic.load_state_dict(ck["critic"])
    self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
    self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
    self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
    self.iteration = int(ck["iteration"])
    self.num_updates = int(ck["num_updates"])
    print(f"[INFO] resumed from {path} at iteration {self.iteration}", flush=True)

  # ------------------------------------------------------------------ loop
  def learn(self, num_iterations: int | None = None, max_hours: float | None = None) -> None:
    cfg = self.cfg
    sch = cfg.schedule
    env = self.env
    total = int(num_iterations or sch.total_iterations)
    hours_cap = sch.max_hours if max_hours is None else max_hours

    # Reset before the first state is read: mjlab's data buffers are zero until the first
    # reset, and a zero quaternion in the replay buffer makes the world-model forward NaN
    # (quat_normalize(0) = 0/0). The reference loop starts with `obs, _ = env.reset()`.
    if hasattr(env, "reset"):
      env.reset()
    obs = self._observation()
    if self.adapter is not None:
      self._on_reset(None)
    s = self._state()
    xi = self._xi()
    frame, clip = self._frame_clip()
    self._prev_s[:] = s
    self._prev_xi[:] = xi
    self._prev_frame[:] = frame
    self._prev_clip[:] = clip

    t_start = time.time()
    timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
    agg: dict[str, float] = {}
    agg_n = 0
    finished_returns: list[float] = []
    finished_lens: list[float] = []
    last_log = time.time()
    env_ids = torch.arange(int(env.num_envs), device=self.device)

    while self.iteration < total:
      # ---------------- (1) collect one vectorised env step
      t0 = time.perf_counter()
      with torch.no_grad():
        collector = self.policy_collect if getattr(self, "ema_tau", 0.0) > 0.0 else self.policy
        a_pol = collector(obs["policy"], obs["goal"])
        if getattr(cfg, "clip_actions", True):
          # the decoder has no output nonlinearity and this runner bypasses the env wrapper
          # that normally clips; unclipped actions (|a| up to 62 observed) throw the robot down
          a_pol = a_pol.clamp(-1.0, 1.0)
        a_env = self._env_action(a_pol)
        # The environment normally applies its own action noise (the reference env does:
        # env.action_noise_std). Adding it here too doubles the effective noise (x1.41) and
        # measurably degrades the data and the failure rate, so the runner only adds noise
        # for adapters that explicitly ask for it.
        noise = float(getattr(cfg.env, "action_noise_std", 0.0)) if self._adds_action_noise else 0.0
        if noise > 0.0:
          a_env = a_env + noise * torch.randn_like(a_env)
        _obs_raw, rew, term, tout, _ = env.step(a_env)
        # observations always come from the adapter (the env's groups may be named
        # differently, e.g. 'actor'/'critic' for the velocity task)
        obs = self._observation()
        done = term | tout
        self.ep_reward_sum += rew
        a_prev = getattr(self, "_a_prev", a_env)
        self._a_prev = a_env.detach()
        self.ep_len += 1
        if __import__("os").environ.get("SM_DEBUG_LOOP") and (
          self.iteration < 6 or self.iteration % 25 == 0 or float(rew.mean()) < -0.5
        ):
          print(
            f"[dbg] it {self.iteration} rew {float(rew.mean()):+.4f} min {float(rew.min()):+.3f} "
            f"term {int(term.sum())} tout {int(tout.sum())} done {int(done.sum())} "
            f"ep_len {float(self.ep_len.mean()):.1f} epsum {float(self.ep_reward_sum.mean()):+.3f} "
            f"|a| {float(a_env.abs().mean()):.4f} max {float(a_env.abs().max()):.4f} "
            f"|da|2 {float((a_env - a_prev).pow(2).sum(-1).mean()):.4f} finite {bool(torch.isfinite(self._prev_s).all())}",
            flush=True,
          )

        # transition rows (state before the step -> state after it). Skipped when the
        # environment records them itself (adapter.writes_buffer == False); two writers
        # into one ring buffer would scramble the sampled windows.
        r_terms = None
        if self._writes_buffer:
          cols = self.heads.learned_columns(env) if hasattr(self.heads, "learned_columns") else ()
          mgr = getattr(env, "reward_manager", None)
          step_reward = getattr(mgr, "_step_reward", None)
          if cols and torch.is_tensor(step_reward):
            r_terms = step_reward.detach()[:, list(cols)].to(torch.float32)
        if self._writes_buffer:
          self.buffer.add(
            env_ids,
            s=self._prev_s,
            a=a_pol,
            a_exec=a_env,
            xi=self._prev_xi,
            r=rew,
            done=term.float(),
            frame=self._prev_frame,
            clip=self._prev_clip,
            terminal=(done.clone() if (self._auto_reset and self._writes_buffer) else torch.zeros_like(done)),
            ep_start=self._ep_start,
          )
        self._ep_start[:] = False

        if done.any():
          ids = done.nonzero().squeeze(-1)
          lens = self.ep_len[ids]
          norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
          finished_returns.extend(norm_ret.tolist())
          finished_lens.extend(lens.tolist())
          self.ep_reward_sum[ids] = 0.0
          self.ep_len[ids] = 0.0
          if self._writes_buffer and not self._auto_reset:
            # terminal row: the state the last action reached (no transition out of it)
            s_next = self._state()
            zeros_a = torch.zeros(ids.numel(), a_env.shape[-1], device=self.device)
            f_now, c_now = self._frame_clip()
            self.buffer.add(
              ids,
              s=s_next[ids],
              a=zeros_a,
              a_exec=zeros_a,
              xi=self._prev_xi[ids],
              r=torch.zeros(ids.numel(), device=self.device),
              done=term[ids].float(),
              frame=f_now[ids],
              clip=c_now[ids],
              terminal=torch.ones(ids.numel(), dtype=torch.bool, device=self.device),
              ep_start=torch.zeros(ids.numel(), dtype=torch.bool, device=self.device),
            )
            self.buffer.end_episode(ids)
          if not self._auto_reset:
            env.reset(env_ids=ids)
          self._on_reset(ids)
          obs = self._observation()

        s2 = self._state()
        xi2 = self._xi()
        f2, c2 = self._frame_clip()
        self._prev_s[:] = s2
        self._prev_xi[:] = xi2
        self._prev_frame[:] = f2
        self._prev_clip[:] = c2
        if done.any():
          self._prev_s[ids] = s2[ids]
          self._ep_start[ids] = True
      timers["sim"] += time.perf_counter() - t0
      self.iteration += 1

      # ---------------- (2)+(3) updates at the configured update-to-data ratio
      need_rows = cfg.wm_trainer.window + cfg.policy_trainer.history + 2
      # depth = rows written per env (the time axis); buffer.size counts envs*rows and is
      # already large after one step, which made a freshly resumed run update immediately
      # and crash sampling a k-step window out of an empty buffer.
      depth = int(self.buffer.ptr.min()) if hasattr(self.buffer, "ptr") else int(self.buffer.size)
      self._windows_ok = True
      # The reward model trains from the very first rows: it has to be usable *before* the
      # policy starts consuming it, otherwise the policy fits pure noise.
      if self.reward_opt is not None and depth >= 2:
        try:
          if not self._reward_scales_set:
            w0 = self.buffer.sample_windows(min(256, depth), 1)
            self.heads.model.set_scales_from(w0.states[0], w0.a_exec[0], w0.xi)
            self._reward_scales_set = True
          for _ in range(self.reward_cfg_rm.updates_per_iteration):
            m_r = self.heads.update(self.buffer, self.reward_cfg_rm, self.reward_opt)
        except RuntimeError:
          m_r = {}
      # the learned reward heads must be fitted before the policy may consume them; until then only
      # the world model and the heads train (an untrained head is noise, measured 14% reward error)
      learned = getattr(self.heads, "learned", None)
      min_corr = float(getattr(getattr(cfg, "learned_cfg", None), "min_corr", 0.9))
      if learned is not None and depth >= 2:
        if not hasattr(self, "_learned_opt"):
          self._learned_opt = torch.optim.Adam(learned.parameters(), lr=float(getattr(getattr(cfg, "learned_cfg", None), "lr", 3e-4)))
        if getattr(self.buffer, "r_terms", None) is not None:
          try:
            m_learned = learned.update(self.buffer, getattr(cfg, "learned_cfg", None), self._learned_opt)
          except RuntimeError:
            m_learned = {}
      learned_ready = True if learned is None else learned.ready(min_corr)
      if learned is not None and not learned_ready and self.iteration % 50 == 0:
        corr = ", ".join(f"{n}={float(learned.corr[k]):+.2f}" for k, n in enumerate(learned.names))
        print(f"[wm-spec] learned reward not ready (min corr {min_corr}): {corr}; "
              f"policy updates are held until it is", flush=True)
      if self.iteration >= sch.warmup_iterations and depth >= need_rows:
        self._utd_acc += sch.updates_per_env_step
        while self._utd_acc >= 1.0:
          self._utd_acc -= 1.0
          lr = cosine_lr(
            self.iteration, total, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction
          )
          self.wm_trainer.set_lr(lr)
          self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
          t0 = time.perf_counter()
          try:
            m_wm = self.wm_trainer.update(self.buffer)
          except RuntimeError as exc:
            if "no sampleable rows" not in str(exc):
              raise
            self._skipped_updates += 1
            if self._skipped_updates in (1, 100, 1000):
              print(f"[warn] no k-step window yet (episodes too short); "
                    f"skipped {self._skipped_updates} update groups", flush=True)
            self._utd_acc = 0.0
            break
          timers["wm"] += time.perf_counter() - t0
          t0 = time.perf_counter()
          start_pi = int(getattr(cfg, "policy_start_updates", 0) or 0)
          if self.num_updates >= start_pi and learned_ready:
            for _ in range(sch.policy_updates_per_wm_update):
              m_pi = self.policy_trainer.update(self.buffer)
              if getattr(self, "ema_tau", 0.0) > 0.0:
                with torch.no_grad():
                  for _pe, _p in zip(self.policy_collect.parameters(), self.policy.parameters()):
                    _pe.mul_(self.ema_tau).add_(_p, alpha=1.0 - self.ema_tau)
          else:
            m_pi = {}
          timers["policy"] += time.perf_counter() - t0
          self.num_updates += 1
          for k, v in m_wm.items():
            agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
          for k, v in m_pi.items():
            agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
          agg["lr"] = lr
          agg_n += 1

      # ---------------- logging / checkpoints
      if self.iteration % sch.log_interval == 0:
        it = self.iteration
        elapsed = time.time() - t_start
        for k, v in agg.items():
          self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
        tr = sum(finished_returns) / len(finished_returns) if finished_returns else float("nan")
        if finished_returns:
          self.writer.add_scalar("train/episode_norm_return", tr, it)
          self.writer.add_scalar("train/episode_len", sum(finished_lens) / len(finished_lens), it)
        with torch.no_grad():
          err = None
          if self.buffer.size > 2048:
            try:
              err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
            except RuntimeError:
              # every episode is shorter than the window: the probe has nothing to sample
              err = None
            if err is not None:
              self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
              self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
        for k, v in timers.items():
          self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
        self.writer.add_scalar("time/hours", elapsed / 3600, it)
        self.writer.add_scalar("time/updates", self.num_updates, it)
        dt_log = time.time() - last_log
        last_log = time.time()
        print(
          f"it {it} | {elapsed / 3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | "
          f"updates {self.num_updates} | train_ret {tr:.3f} (n={len(finished_returns)}) | "
          + (f"rm mse {m_r['reward_mse']:.3f} mae {m_r['reward_mae']:.4f} corr {m_r['reward_corr']:+.2f} | "
             if self.reward_opt is not None and "m_r" in dir() else "") +
          f"wm nse1 {agg.get('wm/nse_1step', float('nan')) / max(agg_n, 1):.4f} "
          f"tbce {agg.get('wm/term_bce', float('nan')) / max(agg_n, 1):.3f} "
          f"tacc {agg.get('wm/term_acc', float('nan')) / max(agg_n, 1):.2f} "
          f"nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
          f"imag_ret {agg.get('policy/imag_return_undisc', float('nan')) / max(agg_n, 1):.3f} "
          f"surv {agg.get('policy/imag_survival_H', float('nan')) / max(agg_n, 1):.3f} | "
          f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s",
          flush=True,
        )
        agg, agg_n = {}, 0
        finished_returns, finished_lens = [], []
        timers = {k: 0.0 for k in timers}
        if hours_cap is not None and (time.time() - t_start) / 3600 > hours_cap:
          print("[INFO] max_hours reached, stopping", flush=True)
          break

      if self.iteration % sch.checkpoint_interval == 0:
        self.save("ckpt_latest.pt")
        self.save(f"ckpt_{self.iteration:07d}.pt")

    self.save("ckpt_latest.pt")
    self.writer.flush()
