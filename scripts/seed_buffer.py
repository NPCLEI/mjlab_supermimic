"""Record a demonstration rollout into a SuperMimic replay-buffer seed.

Rationale: motion tracking starts from a reference motion, so the replay data already covers the
behaviour being asked for. A velocity task provides nothing of the sort -- at initialisation the
buffer holds only standing and falling, so the world model models falling and imagination never
proposes locomotion. This script records a *walking* policy (PPO) into rows that the SuperMimic
buffer can absorb, so the world model has seen the target behaviour.

    python scripts/seed_buffer.py --ppo-ckpt <ckpt> --task Mjlab-Velocity-Flat-Unitree-G1 \
        --steps 20000 --out seeds/ppo_walk.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import tyro  # noqa: E402


def main(
  ppo_ckpt: Path,
  task: str = "Mjlab-Velocity-Flat-Unitree-G1",
  steps: int = 20000,
  num_envs: int = 512,
  out: Path = Path("seeds/ppo_walk.pt"),
  device: str = "cuda:0",
) -> None:
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  from src.tasks.supermimic.adapters.velocity import VelocityAdapter

  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = int(num_envs)
  raw = ManagerBasedRlEnv(env_cfg, device=device, render_mode=None)
  # the PPO policy expects the wrapper's observation normalisation and action clipping
  wrapped = RslRlVecEnvWrapper(raw, clip_actions=True)

  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    from mjlab.rl import MjlabOnPolicyRunner as runner_cls  # type: ignore[no-redef]
  log_dir = Path(ppo_ckpt).parent
  runner = runner_cls(wrapped, load_rl_cfg(task), str(log_dir), device)
  runner.load(str(ppo_ckpt))
  policy = runner.get_inference_policy(device=device)
  print(f"[seed] loaded {ppo_ckpt}", flush=True)

  adapter = VelocityAdapter(raw)
  obs, _ = wrapped.reset()
  rows = {"s": [], "a": [], "xi": [], "r": [], "done": []}
  ids = torch.arange(int(num_envs), device=device)
  for i in range(int(steps)):
    with torch.no_grad():
      a = policy(obs)
    s = adapter.state(raw)
    xi = adapter.xi(raw)
    obs, rew, dones, infos = wrapped.step(a)
    rows["s"].append(s.detach().cpu())
    rows["a"].append(a.detach().cpu())
    rows["xi"].append(xi.detach().cpu())
    rows["r"].append(rew.detach().cpu())
    rows["done"].append(dones.detach().float().cpu())
    if (i + 1) % 1000 == 0:
      print(f"  {i + 1}/{steps} steps, mean reward {float(rew.mean()):+.4f}", flush=True)

  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save({k: torch.stack(v) for k, v in rows.items()}, out)
  print(f"[seed] wrote {out} ({steps} steps x {num_envs} envs)", flush=True)
  wrapped.close()


if __name__ == "__main__":
  tyro.cli(main)
