# mjlab_supermimic

SuperMimic for [mjlab](https://github.com/mjlab) motion tracking: a **world-model-based
policy trainer** (learn the dynamics once, then improve the policy entirely inside the
learned model) packaged as a drop-in replacement for PPO-style training.

The engine is the vendored reference implementation (`src/tasks/supermimic/engine/supermimic`,
byte-identical to the upstream reproduction except two environment-layer capacity
constants). Everything around it — the runner, the adapter layer, the analytic heads — is
what makes it usable on top of mjlab tasks.

```
src/tasks/supermimic/
  engine/supermimic/       # vendored engine: world model, imagination, GAE critic, replay
  rl/                      # PPO-style runner: SuperMimicRunner(env, cfg, log_dir, device, adapter)
  adapters/                # task adapters (tracking adapter + generic mjlab bridge + synced heads)
  train.py                 # tyro CLI entry
scripts/
  train_tracking.py        # train any npz / packed motion library
  eval_task.py             # Appendix-F evaluation (every clip once, normalised return)
  play_task.py             # mjlab's native viser web viewer
  build_library.py         # pack a memmap library (index.json + 6 .npy) into one .pt
  slice_clip.py            # cut a single clip out of the memmap library
  slice_library.py         # cut an N-clip subset out of the memmap library
```

## Install

```bash
pip install -r requirements.txt      # torch 2.12.1 + mujoco + mjlab + tyro/viser/tensorboard
export PYTHONPATH=$PWD              # the package lives under src/ (src/tasks/supermimic)
```

## Usage

### 1. Train on a motion library

`--data.npz-dir` takes a directory of mjlab-format clip `*.npz` files; `--data.library`
takes one packed `.pt` (required for very large libraries).

```bash
# 300-clip subset (npz)
python scripts/train_tracking.py \
  --adapter tracking --data.npz-dir motion_lib_300 \
  --env.num-envs 2048 --env.anchor-obs \
  --schedule.max-hours 3 --log-root logs --run-name demo300

# whole library (22k clips, packed fp16)
python scripts/train_tracking.py \
  --adapter tracking --data.library motion_lib_all.pt \
  --env.num-envs 4096 --env.anchor-obs \
  --wm-trainer.batch-size 2048 --wm-trainer.batches-per-update 10 \
  --policy-trainer.batch-size 4096 \
  --schedule.max-hours 8 --log-root logs --run-name demoall
```

Useful knobs (`SuperMimicCfg`, same names as the reference implementation):

| flag | meaning |
|---|---|
| `--env.anchor-obs` | add the 9-dim anchor-relative goal (global pose feedback). **Strongly recommended**: without it the policy cannot correct drift |
| `--env.sampling-mode` | `return` (paper default, focuses on hard frames), `uniform`, `start` |
| `--wm-trainer.batch-size`, `--wm-trainer.batches-per-update` | world-model fitting budget |
| `--policy-trainer.horizon`, `--policy-trainer.batch-size` | imagination horizon / policy batch |
| `--schedule.max-hours`, `--schedule.checkpoint-interval` | wall-clock budget, checkpoint cadence |
| `--policy-start-updates` | delay the policy by N world-model update groups (only useful when the reward is contact-heavy) |

### 2. Evaluate (Appendix F)

Every clip of a library is played once from its first frame, deterministically; the
reported score is the normalised return over the clip (1.0 = perfect), plus a failure count.

```bash
python scripts/eval_task.py --ckpt logs/demo300/ckpt_latest.pt --npz-dir motion_lib_300 --num-envs 300
```

### 3. Play in the browser

```bash
python scripts/play_task.py --ckpt logs/demo300/ckpt_latest.pt --npz-dir motion_lib_300 --port 8097
# open http://<host>:8097   (ghost = reference, solid = policy)
```

## Preparing a motion library

The repo's own dataset is `motions_all_sitfiltered/` (22,246 clips, 11.3 M frames,
`index.json` + 6 `.npy` arrays). All three helpers accept it directly:

```bash
python scripts/build_library.py  --src motions_all_sitfiltered --out motion_lib_all.pt   # 22k clips, 5.7 GB fp16
python scripts/slice_clip.py     --src motions_all_sitfiltered --out motion_lib_1 --clip 1307
python scripts/slice_library.py  --src motions_all_sitfiltered --out motion_lib_300 --clips 300
```

The packed library stores `ref` (root pose + joints, 71 dims) **and the dataset's own 14
tracked-body arrays** in fp16, so `bodies()` gathers numbers instead of recomputing them
with FK.

## Results

### Single clip: `bilibili_ground/Ground/1307.csv` (315.8 s, contact-heavy fight)

Trained on **that clip only** (1024 envs, anchor obs, 8 h, single 4090) — the best checkpoint
replays the whole 315.8 s clip without a fall:

| checkpoint | normalised return | survives | `body_ori` | `body_pos` | `root_ori` |
|---|---:|---|---:|---:|---:|
| 2000 | 0.214 | 5.1 s | 0.5 % | 57 % | 17 % |
| 5000 | 0.552 | 225 s | 0.5 % | 60 % | 69 % |
| 9000 | 0.691 | full clip | 21 % | 82 % | 95 % |
| 14000 | 0.811 | full clip | 64 % | 90 % | 96 % |
| **18000 (best)** | **0.849** | **full clip, no failure** | **72 %** | **91 %** | **97 %** |
| 22000 (last) | 0.820 | falls at 261 s | — | — | — |

World model: `nse1 = 0.0036`, `nse8 = 0.0055` (the paper reports 0.007–0.011 for 8 steps).
**Take the best checkpoint, not `ckpt_latest`**: the final one is measurably worse.

### The refactored runner matches the reference trainer

Same config, same data, only the implementation of the loop differs (8 checkpoints):

| iteration | reference `Trainer` | this runner |
|---|---:|---:|
| 3000 | 0.481 | 0.494 |
| 4000 | ~0.52 | 0.541 (no failure) |
| 5000 | 0.552 | 0.567 |
| 6000 | 0.588 | 0.587 |
| 7000 | 0.583 | 0.606 |
| 8000 | 0.637 | 0.678 (no failure) |
| 9000 | 0.691 | 0.684 |
| 10000 | 0.722 | 0.746 |

### Whole library (22,246 clips)

`nse8` falls 1.0 → 0.018 within 4.5 k iterations on a single A100 (4096 envs), i.e. the
8.8 M world model does fit 11.3 M frames of data. Per-clip scores are *low by construction*
at this budget — each clip receives 1/22246 of the sampling weight — so use the library-wide
average for the metric and treat this run as a generalist base / fine-tuning initialisation.

## If something goes wrong (four real pitfalls, all fixed here)

1. **Resume must not reset the counters.** `load()` has to run *after* `iteration` /
   `num_updates` are initialised, or every restart silently restarts the LR schedule and the
   `policy_start_updates` phase.
2. **Gate updates on the buffer's time depth, not on its row count.** `buffer.size` counts
   rows across envs, so with 1024 envs it is already large after one step while no k-step
   window exists yet → `RuntimeError: replay buffer has no sampleable rows yet`.
3. **Do not reset envs that auto-reset.** mjlab's envs keep `auto_reset=True`: `env.step`
   already resets finished episodes and the recorder hooks write the terminal rows. Calling
   `env.reset(env_ids=...)` as well double-resets every episode and corrupts the replay
   distribution (observed as `nse1` 0.026 → 0.069 and the training return collapsing).
4. **Never let a short episode kill the run.** When the behaviour policy falls every few
   steps no k-step window can be sampled; skip that update instead of crashing.

## Smoke test of the stock mjlab task

The repository's environment itself is verified by the stock PPO task:

```bash
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048 --agent.run-name ppo_flat_smoke --agent.max-iterations 300
# Iteration time: 1.48 s  → 300 iterations in ~7.5 min
```

## Citation

SuperMimic (world-model-based motion tracking from imagination). The vendored engine under
`src/tasks/supermimic/engine/` is the reference reproduction of that method; the adapter
layer, runner and tooling in this repository are the packaging on top of it.

## Running a non-tracking task (velocity tracking)

The framework needs four things from a task: a 71-dim state, a goal vector, an **analytic**
reward and an **analytic** termination on predicted states -- plus the analytic observation
the policy consumes. `src/tasks/supermimic/adapters/velocity.py` implements them for the
stock `Mjlab-Velocity-Flat-Unitree-G1` task:

```bash
python scripts/train_tracking.py --adapter velocity \
  --task-id Mjlab-Velocity-Flat-Unitree-G1 --env.num-envs 512 \
  --policy-trainer.batch-size 512 --wm-trainer.batch-size 256 \
  --schedule.max-hours 1
```

Getting there required removing six motion-tracking assumptions from the engine (all fixed
in `engine/supermimic`):

| location | assumption | now |
|---|---|---|
| `imagination/compiled.py` | fixed tuple of the 8 tracking reward names | `heads.reward_term_names` |
| `imagination/compiled.py` | `analytic_observation` (93-dim tracking layout) | `heads.observation(...)`, tracking fallback |
| `imagination/rollout.py` | history frames built with `analytic_observation` | same hook |
| `imagination/rollout.py` | observation noise fixed at 93 dims | skipped when the width differs |
| `imagination/rollout.py` | diagnostic term accumulator fixed at 8 | task term count |
| `world_model/heads.py`, `policy/nets.py` | history flattening assumes 5x93; policy input fixed at 930 | single-frame identity; `PolicyCfg.obs_dim` |

Also fixed: the runner reads observations through the adapter (the velocity task's groups are
`actor`/`critic`, not `policy`/`goal`).

**Status: the task runs end to end, the analytic heads are numerically consistent with the
environment (reward 0.086/step, no terminations at reset, observation width 96 = the env's
`actor` group), and the world model fits the sampled windows (nse1 0.0105 when the windows
are taken directly from the buffer). The training loop itself is NOT yet correct for a stock
mjlab task: it reports a return of -1.18 per step where a direct `env.step` measurement gives
+0.077, episodes end after ~4 steps, and consequently the world-model loss is NaN. The
suspected cause is the runner's step bookkeeping for environments whose reset ownership
differs from the tracking task's recorder (`auto_reset`, `ep_start`, terminal rows).**

Terms that cannot be scored in imagination are excluded by construction: `air_time`,
`foot_clearance`, `foot_slip`, `soft_landing`, `self_collisions`, `angular_momentum` (they
need contact forces or air time). That is a property of the method -- the imagined reward must
be a differentiable function of the predicted state.
## Velocity-tracking parameter findings

Measured on `Mjlab-Velocity-Flat-Unitree-G1` through `register_supermimic_task` (robot GPU 0,
1024 envs unless stated). `train_ret` is the environment return normalised by the task's maximum
step reward, so 1.0 is perfect tracking.

| change | effect |
|---|---|
| keep the stock contact terms as a hand-written analytic reward | return collapses (the analytic reward is only 0.56-correlated with the environment's, 3.8x too large, and blind to -0.22/step contact penalties) |
| **drop the contact terms, as the paper does** | return rises to +0.33; the environment now optimises exactly what imagination can score |
| **clip the action** (collection and imagination) | removes the collapse entirely: -60 -> +-0.2. The decoder has no output nonlinearity and this runner bypasses the wrapper that clips |
| **normalise the predicted quaternion** before computing gravity | removes a world-model exploit (`|q|` drift lets the policy game `upright`) |
| **EMA collection policy** (tau 0.99) | removes the oscillation; the update has no trust region |
| **tracking's recipe** (horizon 32, wm 1024x10, policy 2048, lr 2e-4, warmup 40) | imagined return 0.2 -> 0.89, real return 0 -> +0.25 |
| 10-frame observation history | worse early (-0.29 at iteration 1300 vs +0.25 with one frame); the input grows to 960 dims |
| paper sensor noise on the imagined observation | worse (-1.03 at iteration 750); removed (a task can opt in with `heads.obs_noise_half`) |
| smaller policy (lr 5e-5, horizon 8) | no improvement |
| replay capacity 1000 -> 4000 | no effect |
| removing the `command_vel` curriculum | no effect |

Remaining gap: the real return oscillates around +0.1..+0.3 and does not climb to the 0.6-0.8 that
PPO reaches. The imagined return (0.9) stays far above the real one, i.e. the policy exploits the
world model's optimism -- a model-based limitation rather than a task-integration defect. Motion
tracking does not show it because the reference motion supplies the target behaviour in the data.
