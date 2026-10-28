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
## Whole-library results (22,246 clips, 8 h, one A100)

Trained on the entire `motions_all_sitfiltered` library (22,246 clips / 11.3M frames / 313 hours),
packed as `motion_lib_all.pt` (ref 71 dims + the dataset's own 14 body arrays, fp16, 5.74 GB).

| evaluation set | normalised return | failures |
|---|---:|---:|
| 300-clip in-distribution subset, after 7.4 h (best ckpt 22000) | **0.8248** | 8/300 |
| same, final ckpt 24000 | 0.8215 | 10/300 |
| 6 long fall-and-get-up clips (from the 125 fall clips in the library) | **0.6703** | 6/6 |

Reference points: a model trained on *only* those 300 clips for 3 h reaches 0.7156, and a model
trained on a single clip reaches 0.849 on that clip alone. So the generalist beats the 300-clip
specialist by 15% while covering 74x more motions, and rare classes (fall/get-up is 0.6% of the
library) are its clear weak spot: 0.670 vs 0.825, and every one of those six clips fails partway.

World model on this run: `nse1 0.0047`, `nse8 0.0094` (paper reports 0.007-0.011 for 8 steps).

Two practical notes that cost real time:

* **enable TF32 on an A100.** Its fp32 peak (19.5 TFLOPS) is four times *below* an RTX 4090 (82.6),
  so plain fp32 gains nothing from the bigger card. With `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` the
  per-sample update cost drops from 1.44 ms to 0.61 ms, which is what lets the policy batch double.
* **pack the dataset's body arrays, do not recompute them.** The compact format alone stores only
  `ref`, and `bodies()` then runs differentiable FK for every imagined step.

Rare classes need resampling: sampling is uniform over clips, so a class that is 0.6% of the library
never gets enough samples regardless of `sampling_mode=return` (which weights by return, not by
class frequency).
## 速度追踪：为什么 SuperMimic 学不出来（实测结论）

同一个框架、同一份代码，在动作追踪（tracking）上能到 0.849，在速度追踪上只能学到"站着不动"。这不是接入问题——接入层面的 10 个 bug 全部定位并修复了——而是方法前提的差异。

### 一、接入层面修掉的 10 个真实缺陷

| # | 缺陷 | 症状 |
|---|---|---|
| 1 | `acos(clamp(-g_z,-1,1))` 在边界求导发散 | 损失有限但**世界模型梯度全 NaN** |
| 2 | `auto_reset` 归属冲突（env 与 runner 都复位）| 终止行写的是复位后状态 |
| 3 | 世界模型状态规范化尺度按 tracking 设定 | `out_scale=0.02` 表达不了 12 rad/s 的关节速度 |
| 4 | 手写解析奖励与环境奖励不一致 | 相关系数仅 0.56、均值差 3.8 倍、看不见 −0.22/步的接触惩罚 |
| 5 | 想象批 4096 与实环境 512 的 goal 未按采样行对齐 | 形状错 |
| 6 | **采样时动作未限幅** | 动作输出到 62，回报崩到 −60 |
| 7 | **想象 rollout 里动作未限幅** | 想象回报 −537 |
| 8 | **预测四元数未归一化** | `upright`/终止判据可被"缩小四元数"骗过 |
| 9 | 更新无信赖域 | 一批坏梯度就把策略推飞（加 EMA 行为策略解决）|
| 10 | 引擎 6 处 tracking 硬编码 | 只能训 tracking 类任务 |

按论文做法把 8 个接触/姿态类奖励项从环境中去掉后，环境奖励与解析奖励完全一致；动作限幅后灾难性崩溃消失。

### 二、参数结论（全部实测）

| 参数 | 结论 |
|---|---|
| 接触项奖励 | 必须去掉（保留则必崩）|
| 动作限幅（采样 + 想象）| 必须 |
| 四元数归一化 | 必须 |
| EMA 行为策略 | 必须（否则震荡）|
| tracking 那套（horizon 32、WM 1024×10、policy 2048、lr 2e-4、warmup 40）| 显著有效（想象回报 0.2→0.89）|
| anchor 等价物（速度误差 4 维输入）| 有效（集长从 0.4 s → 3.4 s）|
| 10 帧观测历史 | 有害（输入 960 维，−0.29 vs +0.25）|
| 论文传感器噪声 | 有害（−1.03）|
| lr 5e-5 / horizon 8 / 回放容量 ×4 / 去掉课程 | 无效 |

### 三、决定性测量

训练 2.45 小时后（`nse1 0.0014`，世界模型很准）在真实环境里测：

```
指令 0.8 m/s 前进，400 步，64 envs
mean |v − 0.8| = 0.816 m/s        ← 误差 ≈ 指令本身，即根本没动
中位数           0.784
±0.2 m/s 内占比  1.2%
mean |yaw rate|  0.430 rad/s      ← 在原地转
episode 平均 3.09 s
```

`train_ret +0.30` 全部来自"站着不动"：`upright`（权重 1.0）满 + 动作平稳，两项追踪奖励只有 `exp(-0.8²/0.25) ≈ 0.02`。

世界模型对动作的敏感度（同状态同动作，+0.5 推一次）：

```
真实：dv = [-0.005, +0.003, -0.199]
模型：dv = [-0.003, +0.006, -0.022]
比值 = 0.116                        ← 模型只"看见"12% 的动作效果
想象奖励：站住 −0.1042 → 推一下 −0.1522（Δ −0.048）
```

### 四、结论

**"站着不动"是想象与真实环境共同的局部最优**：`upright` 给满分、追踪项因不动而接近 0、而世界模型对动作的响应只有真实的 12%，所以在想象里"迈步"看不出收益。策略于是停在站立。

tracking 之所以成功，是因为**参考动作把机器人推在正确的状态流形上**，奖励是围绕已知轨迹的密核；速度追踪需要"从零探索出步态"，而世界模型的训练数据里从来没有走路（初始化时只有站与摔）。

**这不是复现出了问题，而是方法前提的差异：SuperMimic 是"由示范驱动的位姿追踪"，不是"从零探索的强化学习"。**

### 五、要让速度追踪跑出来，需要改变前提（三选一）

1. **给示范**：先用 PPO 训出会走的策略，把 rollout 灌进回放缓冲（仓库内 `scripts/seed_buffer.py` 已就绪）
2. **改奖励地形**：把 `upright` 权重降到 0.1、追踪项权重提高，让"站着"不再是高点（纯参数）
3. **换探索**：在真实环境里做探索（本质上是 PPO），即放弃纯 model-based
## 速度追踪：为什么 SuperMimic 学不出来（实测结论）

同一个框架、同一份代码，在动作追踪（tracking）上能到 0.849，在速度追踪上只能学到"站着不动"。这不是接入问题——接入层面的 10 个 bug 全部定位并修复了——而是方法前提的差异。

### 一、接入层面修掉的 10 个真实缺陷

| # | 缺陷 | 症状 |
|---|---|---|
| 1 | `acos(clamp(-g_z,-1,1))` 在边界求导发散 | 损失有限但**世界模型梯度全 NaN** |
| 2 | `auto_reset` 归属冲突（env 与 runner 都复位）| 终止行写的是复位后状态 |
| 3 | 世界模型状态规范化尺度按 tracking 设定 | `out_scale=0.02` 表达不了 12 rad/s 的关节速度 |
| 4 | 手写解析奖励与环境奖励不一致 | 相关系数仅 0.56、均值差 3.8 倍、看不见 −0.22/步的接触惩罚 |
| 5 | 想象批 4096 与实环境 512 的 goal 未按采样行对齐 | 形状错 |
| 6 | **采样时动作未限幅** | 动作输出到 62，回报崩到 −60 |
| 7 | **想象 rollout 里动作未限幅** | 想象回报 −537 |
| 8 | **预测四元数未归一化** | `upright`/终止判据可被"缩小四元数"骗过 |
| 9 | 更新无信赖域 | 一批坏梯度就把策略推飞（加 EMA 行为策略解决）|
| 10 | 引擎 6 处 tracking 硬编码 | 只能训 tracking 类任务 |

按论文做法把 8 个接触/姿态类奖励项从环境中去掉后，环境奖励与解析奖励完全一致；动作限幅后灾难性崩溃消失。

### 二、参数结论（全部实测）

| 参数 | 结论 |
|---|---|
| 接触项奖励 | 必须去掉（保留则必崩）|
| 动作限幅（采样 + 想象）| 必须 |
| 四元数归一化 | 必须 |
| EMA 行为策略 | 必须（否则震荡）|
| tracking 那套（horizon 32、WM 1024×10、policy 2048、lr 2e-4、warmup 40）| 显著有效（想象回报 0.2→0.89）|
| anchor 等价物（速度误差 4 维输入）| 有效（集长从 0.4 s → 3.4 s）|
| 10 帧观测历史 | 有害（输入 960 维，−0.29 vs +0.25）|
| 论文传感器噪声 | 有害（−1.03）|
| lr 5e-5 / horizon 8 / 回放容量 ×4 / 去掉课程 | 无效 |

### 三、决定性测量

训练 2.45 小时后（`nse1 0.0014`，世界模型很准）在真实环境里测：

```
指令 0.8 m/s 前进，400 步，64 envs
mean |v − 0.8| = 0.816 m/s        ← 误差 ≈ 指令本身，即根本没动
中位数           0.784
±0.2 m/s 内占比  1.2%
mean |yaw rate|  0.430 rad/s      ← 在原地转
episode 平均 3.09 s
```

`train_ret +0.30` 全部来自"站着不动"：`upright`（权重 1.0）满 + 动作平稳，两项追踪奖励只有 `exp(-0.8²/0.25) ≈ 0.02`。

世界模型对动作的敏感度（同状态同动作，+0.5 推一次）：

```
真实：dv = [-0.005, +0.003, -0.199]
模型：dv = [-0.003, +0.006, -0.022]
比值 = 0.116                        ← 模型只"看见"12% 的动作效果
想象奖励：站住 −0.1042 → 推一下 −0.1522（Δ −0.048）
```

### 四、结论

**"站着不动"是想象与真实环境共同的局部最优**：`upright` 给满分、追踪项因不动而接近 0、而世界模型对动作的响应只有真实的 12%，所以在想象里"迈步"看不出收益。策略于是停在站立。

tracking 之所以成功，是因为**参考动作把机器人推在正确的状态流形上**，奖励是围绕已知轨迹的密核；速度追踪需要"从零探索出步态"，而世界模型的训练数据里从来没有走路（初始化时只有站与摔）。

**这不是复现出了问题，而是方法前提的差异：SuperMimic 是"由示范驱动的位姿追踪"，不是"从零探索的强化学习"。**

### 五、要让速度追踪跑出来，需要改变前提（三选一）

1. **给示范**：先用 PPO 训出会走的策略，把 rollout 灌进回放缓冲（仓库内 `scripts/seed_buffer.py` 已就绪）
2. **改奖励地形**：把 `upright` 权重降到 0.1、追踪项权重提高，让"站着"不再是高点（纯参数）
3. **换探索**：在真实环境里做探索（本质上是 PPO），即放弃纯 model-based
