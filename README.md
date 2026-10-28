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
