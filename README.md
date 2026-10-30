# SuperMimic in mjlab

A reproduction of **[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model](https://arxiv.org/abs/2610.28480)** (CoRL 2026) on the Unitree G1 in
[mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp — plus a rework of how a SuperMimic task is
*declared*, so that adding a task is a two-line registration rather than a new subclass.

<p align="center">
  <img src="docs/assets/fig1_1307_benchmark.png" width="640">
</p>

| | |
|---|---|
| **Single motion (clip 1307, 315.8 s)** | **0.849** normalised return, **15,792 / 15,792 steps, zero failures** — 8 h on one RTX 4090 |
| **Whole library (22,246 clips, 313 h)** | **0.8248** on a held-out 300-clip subset (8 failures), beating a 300-clip specialist (0.7156); 0.6703 on the six fall/get-up clips |
| **World model** | 1-step normalised error **0.0035–0.0047** (paper: 0.007–0.011) |
| **Task/RLO separation** | reward + termination reflected from `env.cfg`; analytic path **equal to the legacy implementation to 3.7e-9** |
| **Velocity tracking** | Appendix-B encoder implemented; from-scratch and encoder variants plateau (§6). A community reproduction reports 652/672 commanded conditions |

---

## 1. What this repository adds

The method is the paper's: one MLP world model `s_{t+1} = f(s_t, a_t, ξ)` over the 71-dim state
(`qpos 36 | qvel 35`), differentiable analytic reward and termination heads, and a motion-tracking
policy trained by backpropagating through 32 imagined steps (SHAC-style, no trust region), with a small
critic bootstrapping the tail and a UTD (update-to-data) ratio of 0.4.

The contribution here is **how the task is declared**, which is what makes the method usable for tasks it
was not written for:

```python
register_mjlab_task(
    task_id="Mjlab-EE-Tracking-Unitree-G1",
    env_cfg=unitree_g1_motions_all_tracking_env_cfg(),        # rewards, terminations, obs, commands
    play_env_cfg=unitree_g1_motions_all_tracking_env_cfg(play=True),
    rl_cfg=supermimic_runner_cfg(                             # training hyper-parameters only
        world_model=WorldModelCfg(input_norm=True),
        policy_trainer=PolicyTrainerCfg(batch_size=4096, horizon=32, lr=2e-4),
        schedule=ScheduleCfg(max_hours=8, warmup_iterations=40),
        strict=True,
    ),
    runner_cls=SuperMimicOnPolicyRunner,
)
```

`rl_cfg` never names a reward, a termination, an observation or a task. Everything task-specific comes
from `env_cfg` through `build_wm_spec(env, cfg)`, which reads `env.cfg` and classifies every non-zero
reward term into one of three channels:

| channel | when | what happens |
|---|---|---|
| **analytic** | the term name has a differentiable projection registered in `adapters/terms/registry.py` (8 tracking terms, 6 command terms) | evaluated exactly on the predicted state — verified equal to the environment's own value |
| **learned** | no projection, but the environment computes the term (contacts, sensors) | one small MLP per term, supervised by the environment's *own* per-term reward, gated at `corr ≥ 0.9`, output clamped to `mean ± 5σ`; before the gate is met **policy updates are held** instead of training on a proxy |
| **unavailable** | neither | dropped loudly; with `strict=True` the run refuses to start |

Startup always prints the classification:

```
[wm-spec] analytic 8 | learned 0 | dropped 0 | max_step_reward 0.1000
```

and the analytic path reproduces the legacy implementation **term by term**:

```
term                          legacy          spec      max|diff|
motion_global_root_pos    +0.00102955   +0.00102955   0.0e+00
motion_body_pos           +0.01261960   +0.01261960   9.3e-10
motion_body_ang_vel       +0.01371508   +0.01371508   0.0e+00
action_rate_l2            -0.00937071   -0.00937071   0.0e+00
total                     +0.03265205   +0.03265205   3.7e-09
observation 0.0   termination features 0.0   terminated agrees: True
```

---

## 2. Results

### 2.1 One motion: clip 1307 (315.8 s), 1x4090, 1024 envs, 8 h

| checkpoint | return | steps completed | failures |
|---|---:|---:|---:|
| 18,000 | **0.849** | 15,792 / 15,792 | 0 |
| 19,000 | 0.835 | 15,792 / 15,792 | 0 |
| 22,000 | 0.820 | fails at 261 s | 1 |

Per-term breakdown of the 0.849 checkpoint: body orientation 72 %, body position 91 %, root position
81 %, root orientation 97 %, linear velocity 96 %, angular velocity 80 %. World model `nse1 0.0036`,
`nse8 0.0055`. **Always take the best checkpoint**: the return peaks and then degrades on long clips.

### 2.2 Whole library: 22,246 clips / 11.34 M frames / 313 h

Packed to 5.74 GB fp16 (including per-clip body arrays), built in 11.6 s. Trained on one A100 80 GB,
4096 envs, world model 2048×10, policy 4096, 8.0 h / 25,400 iterations.

<p align="center">
  <img src="docs/assets/fig3_whole_library.png" width="520">
</p>

| evaluation | return | failures |
|---|---:|---:|
| 300-clip held-out subset, generalist (ckpt 22,000) | **0.8248** | 8 / 300 |
| same subset, per-clip specialist | 0.7156 | — |
| six fall/get-up clips, generalist | 0.6703 (step-weighted 0.5829) | 6 / 6 |
| paper, BONES-SEED, 3 h / 1xH100 | > 0.90 | 25 / 129,514 |

The generalist beats the specialist on the same 300 clips, which is the paper's central claim
(compute grows slowly with library size). Fall/get-up is the weak class: 125 of the 22,246 clips (0.6 %)
contain a fall, and the paper only drives that class to zero failures after 9.8 h on two GPUs.

### 2.3 Input normalisation (A/B, same config, one variable)

Running mean/variance on state features, actions and ξ, updated from real transitions only, inside the
world model (`WorldModelCfg.input_norm`).

<p align="center">
  <img src="docs/assets/fig2_world_model_error.png" width="560">
</p>

| iteration | baseline | **+ input normalisation** |
|---|---:|---:|
| `nse1` @ 3,000 | 0.0238 | **0.0098** (2.4x) |
| `nse1` @ 4,400 | — | 0.0036 (baseline needed 18,000) |
| benchmark @ 5,000 | 0.552 (fails at 225 s) | **0.561 (full clip)** |
| benchmark @ 7,000 | 0.583 | **0.619** |
| benchmark @ 10,000 | 0.722 | **0.750** |

The same conclusion appears independently in the community reproductions (§7): world-model
normalisation is "essential for stable training" (Breezewrf), and per-module input normalisation is
standard practice (vibed, Supermimic_mod).

### 2.4 Which reward terms a learned head can recover

<p align="center">
  <img src="docs/assets/fig5_learned_rewards.png" width="600">
</p>

Measured by regressing the environment's own per-term rewards: only `angular_momentum` reaches the 0.9
gate; `pose`, `foot_clearance`, `soft_landing`, `foot_slip`, `foot_swing_height` and `self_collisions`
plateau at 0.5–0.85, because they depend on contact and phase rather than on `(s, a, ξ)`. They are
therefore **removed from the reward on both sides** (environment and imagination), which is exactly what
the `-SM` task does by default and what the paper does by dropping the self-collision term.

---

## 3. Repository layout

```text
src/tasks/supermimic/
├── rl/
│   ├── spec.py              # WmTermSpec / RewardCtx / SpecReport — the task contract
│   ├── runner.py            # collection, WM fitting, imagination, policy/critic updates
│   ├── mjlab_runner.py      # mjlab runner adapter (asdict rl_cfg, load/save/git log)
│   └── config.py            # SuperMimicCfg: training hyper-parameters only
├── adapters/
│   ├── from_env.py          # build_wm_spec(env, cfg): reflect env.cfg into the contract
│   ├── state.py             # EngineStateProvider / MjlabStateProvider
│   ├── terms/               # analytic families (tracking, command), learned heads, registry
│   ├── tracking.py          # engine-native motion-tracking environment
│   └── velocity*.py         # command tasks, Appendix-B velocity encoder
├── configs/example_tasks.py # register_mjlab_task(...) examples
├── engine/                  # vendored SuperMimic engine (world model, heads, imagination)
├── data/                    # npz -> packed library, slicing, manifests
scripts/                     # train / play / evaluate / build_library / seed_buffer
docs/lessons.md               # the full experiment log: what was measured and what failed
docs/RESULTS.md               # retrospective with every number and date
```

---

## 4. Quickstart

```bash
# single clip: 8 h, one GPU
python scripts/train_tracking.py --adapter tracking --data.npz-dir /path/to/motion_lib_1 \
  --env.num-envs 1024 --env.episode-length-s 20 --env.anchor-obs \
  --schedule.max-hours 8 --run-name one

# whole library (pack it first, then train on the pack)
python scripts/build_library.py --npz-dir /path/to/motions --out data/library_motion
python scripts/train_tracking.py --adapter tracking --data.library data/library_motion \
  --env.num-envs 4096 --env.anchor-obs --schedule.max-hours 8 --run-name library

# paper Appendix-F benchmark (every clip once, deterministic) and interactive replay
python scripts/eval_benchmark.py --ckpt logs/one/ckpt_0018000.pt --npz-dir /path/to/motion_lib_1
python scripts/play.py --ckpt logs/one/ckpt_0018000.pt --npz-dir /path/to/motion_lib_1 --num-envs 4
python scripts/play.py --ckpt logs/one/ckpt_0018000.pt --npz-dir /path/to/motion_lib_1 --clip-index 1307
```

The engine's own entry point (`python -m supermimic.scripts.train ...`, with `PYTHONPATH` pointing at
`src/tasks/supermimic/engine`) takes the same flags; `scripts/train_tracking.py` is the mjlab-side wrapper.
Long jobs: use `systemd-run --user --unit=...`; tmux and `setsid` die with the ssh cgroup.

---

## 5. Reproducing on limited hardware

| budget | setting | outcome |
|---|---|---|
| 1x4090 24 GB | 1024 envs, world model 1024×8, policy 2048 | 0.750 at 10k iterations (~4 h); 0.849 at 18k (~8 h) |
| 1xA100 80 GB | 4096 envs, world model 2048×10, policy 4096 | whole library in 8 h |
| 1xA100 + TF32 | `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` | 1.44 → 0.61 ms per-sample update |

---

## 6. What does *not* work: velocity tracking

The paper evaluates a velocity controller (Sec. 5.5 / Appendix B): a **new encoder**
`z = enc_v(proprio_history_930, command_3)` feeding the **motion-tracking decoder**, so the controller
inherits the gait instead of exploring locomotion.

This repository implements that interface (`Policy.encoders["velocity"]`, 930-dim engine-layout history,
3-dim command scaled by `[2.0, 1.32, 3.35]`, 672-condition step test). Measured outcomes:

| configuration | commands passed | observation |
|---|---:|---|
| PPO in the simulator (4096 envs, 4,000 iters) | 12.2 % | 2.15 s episodes |
| SuperMimic trained from scratch (3.06 h, 62k iters) | 6–10 % | learns to *balance* (episodes 2.07 → 4.97 s), not to track speed |
| Appendix-B encoder, decoder **frozen** | 0 / 64 | falls at 0.66 s — a random encoder emits latents outside the decoder's training distribution |
| Appendix-B encoder, encoder+decoder **joint** | 0 / 64 | stands for full episodes, tracks nothing |
| + world model warm-started from the tracking run | collapses (`nse1 nan`) | the two tasks use incompatible ξ layouts |
| + replay buffer seeded with recorded walking | collapses | seeded segments are discontinuous with timeout resets |

<p align="center">
  <img src="docs/assets/fig4_velocity.png" width="600">
</p>

The measured ceiling is the world model's **action sensitivity, 11.6 % of the simulator's**
(`real dv [-0.005, +0.003, -0.199]` vs `wm dv [-0.003, +0.006, -0.022]`), so imagined stepping earns
almost nothing while the upright term already pays in full — standing is the shared local optimum. The
reference reproduction that reports 652/672 commands trains the velocity encoder on a world model that
has already seen walking data from the motion-tracking phase; reproducing that requires one world model
spanning both stages, which this repository does not yet do.

Two measurement errors were made and fixed here, and are worth recording: the error must be taken
against the command the environment **actually issued** (it resamples ±1 m/s, so assuming 0.8 m/s made
`|v − 0.8|` meaningless), and each checkpoint must be evaluated on the **same seeded command sequence**
(one checkpoint measured 8 % or 16 % depending on the draw).

---

## 7. Community reproductions

Independent reproductions and what they report, as of October 2026. Only one reports a velocity result.

| repository | data | result | velocity |
|---|---|---|---|
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (716★) | 40 LAFAN1 | 0.829, 0/40 failures, 1.47 h on RTX PRO 6000 | no |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (315★) | BONES-SEED 129,491 | **0.930**, 145 failures (0.11 %), 7.1 h on 4x5090 | explicitly not implemented |
| [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe) | 40 LAFAN1 | 0.833 at 1.34 h; best 0.947, 0/40 | **652/672 commands** |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | 40 LAFAN1 | 0.8528, 40/40 | implemented, not converged |
| [YanYuan98/Supermimic_mod](https://github.com/YanYuan98/Supermimic_mod) | 40 LAFAN1 | >0.80 with anchor observations | not implemented |
| **this repository** | 22,246 clips / 313 h | 0.8248 on a 300-clip subset; 0.849 single clip | implemented, plateau (§6) |

Two engineering findings are consistent across all of them and match this repository:

* **Anchor-relative observations matter on long clips.** CuboiLeo reports that pure proprioception drifts
  past 15° of yaw within ~9 s on 100–300 s clips, capping the whole-clip return at ~0.58; adding
  `motion_anchor_ori_b` lifts it to 0.904. YanYuan98 measures the same 0.56 → 0.80+. Our 0.849 on a
  315.8 s clip uses `--env.anchor-obs`.
* **Input normalisation** is reported as essential (Breezewrf), and measured here as a 2.4x faster world
  model with a 2–6 % higher return (§2.3).

### 7.1 Capability matrix

`✓` present, `~` partial, `–` absent, as stated by each project's own documentation.

| | **this repo** | vibed (716★) | CuboiLeo (315★) | dhmbb2 | Breezewrf | Supermimic_mod |
|---|---|---|---|---|---|---|
| Self-built motion library | **✓ 22,246 clips / 313 h** | – (40 LAFAN1) | – (BONES-SEED 129k) | – (40 LAFAN1) | – (40 LAFAN1) | – (40 LAFAN1) |
| Whole-library generalisation measured | **✓ generalist > specialist** (0.8248 vs 0.7156) | – | – | – | – | – |
| Longest evaluated single clip | **315.8 s → 0.849, 0 failures** | ~300 s (40 clips) | 100–300 s → 0.904 (anchor) | ~300 s → 0.947 | ~300 s → 0.8528 | ~300 s → 0.80+ |
| Fall/get-up class reported | **✓ 0.6703 (6/6 fail, honest)** | ~ (clips included) | ✓ (acrobatics analysed) | ~ | – | – |
| Task declared without touching the trainer | **✓ `register_mjlab_task(..., rl_cfg)`** | – | – | – | – | – |
| Reward/termination contract verification | **✓ 3.7e-9 term-by-term vs legacy** | – | – | – | – | – |
| Learned rewards for non-analytic terms | **✓ supervised + `corr ≥ 0.9` gate + measured drop table** | – | – | – | – | – |
| `strict` mode (refuse to start on unevaluable terms) | **✓** | – | – | – | – | – |
| Input-normalisation A/B with a full curve | **✓ 2.4x @3k, +2–6 % return** | – | – | – | ✓ (ablation) | ~ (adopted) |
| Velocity controller result | – (4 configs, 0/64) | – | – | **✓ 652/672** | – (implemented, not trained) | – |
| Multi-GPU data parallel | – | – | **✓ 4×5090, 91 % scaling** | – | – | ✓ |
| Released weights | – | – | **✓ 223 MB** | ✓ | **✓ policy + WM** | – |
| Ablations beyond the paper | ~ | **✓ critic/H, push recovery** | – | **✓ H, critic, push, BPTT vs PPO** | ✓ (3 factors) | ~ (bf16, critic features) |
| Render / viewer tooling | ✓ viewer + clip selector | **✓ render + video tools** | ✓ media | ✓ render rollouts | ✓ MuJoCo play | ✓ video vs RGMT |
| Tests | ✓ 	ests/test_reward_equivalence.py (contract proof, 3.7e-9) | – | ✓ | ✓ | ✓ | **✓ 37 tests** |



### 7.2 Where this repository is genuinely ahead, and where it is not

**Ahead**

1. **Task generality is structural, not documented.** Every other project adds a task by writing a new
   env/adapter path. Here a task is a `register_mjlab_task` call: rewards and terminations are reflected
   from `env_cfg`, `rl_cfg` carries only hyper-parameters, and `strict=True` refuses to start when a
   reward term cannot be evaluated in imagination. The claim is checkable — the analytic path is equal to
   the legacy implementation to `3.7e-9` on a real transition (§1).
2. **The largest measured library, with generalisation rather than a single-clip number.** 22,246 clips
   / 313 h packed to 5.74 GB, and the generalist beats a per-clip specialist *on the same 300 clips*
   (0.8248 vs 0.7156) — the paper's central claim reproduced with an honest held-out protocol. No other
   project reports a library of this size that it built itself.
3. **Reward terms are either exact or provably improvable.** The learned channel is supervised by the
   environment's own per-term rewards, gated at `corr ≥ 0.9`, and the correlation of every term is
   published (§2.4) so the drop decision is auditable. Others either implement a term directly or leave
   it out without measurement.
4. **Negative results are measured, not asserted.** Velocity tracking ships with four configurations,
   the exact reason each fails, and the measured ceiling (world-model action sensitivity 11.6 % of the
   simulator), including two of this project's own measurement errors and their fix (§6).

**Not ahead**

1. **Velocity tracking**: dhmbb2 reports 652/672 commanded conditions; this repository has none (§6).
2. **Scale of the benchmark**: CuboiLeo reports BONES-SEED with 129,491 clips, 0.930 and 0.11 % failures;
   our library is a fifth of that and our failure counts are not directly comparable.
3. **Wall-clock efficiency**: vibed reaches 0.829 with 0/40 in 1.47 h on one RTX PRO 6000; the best
   single-clip run here takes 8 h on a 4090 (a slower card, but the comparison stands).
4. **Multi-GPU and released weights**: CuboiLeo shows 91 % scaling on 4×5090 and publishes a checkpoint;
   this repository is single-GPU and ships no weights.
5. **Ablation breadth**: dhmbb2 covers H ∈ {16, 32, 64, 128}, with/without critic, push recovery and
   BPTT-vs-PPO; this repository runs the normalisation A/B and the reward-channel study only.

---

### 7.3 Head-to-head with [`supermimic_mjlab`](https://github.com/Breezewrf/supermimic_mjlab)

This is the closest comparison in the list: both are mjlab / MuJoCo Warp reproductions of the same paper
on the same robot, so the differences are design choices rather than stack differences.

| | **supermimic_mjlab** | **this repository** | edge |
|---|---|---|---|
| Robot / physics | mjlab PD control, vendored RoboJuDo-Plus G1 (`g1_29dof_rev_1_0.xml`), self-collisions off | mjlab PD control, mjlab G1 29 DoF, self-collisions off | tie |
| Training scale for the headline result | 4096 envs, 15,000 iterations ≈ **4 h on one A6000 48 GB** (FP32) | 1024 envs, 18,000 iterations ≈ **8 h on one RTX 4090** (24 GB) | **theirs** (faster, bigger card) |
| Headline benchmark | 40 LAFAN1 clips (2.45 h): **0.8528, 40/40, 100 % completion** | one 315.8 s clip: **0.849, 15,792/15,792 steps, 0 failures**; 300-clip subset: **0.8248** | **theirs** on breadth, **ours** on library scale |
| Motion data | 40 clips (LAFAN1 retargets) | **22,246 clips / 313 h**, self-built pack (5.74 GB, built in 11.6 s) | **ours** |
| How a task is declared | `config/task_cfg.py` + `rl_cfg.py` (`OnlineConfig`) as plain dataclasses; task code lives in `env.py`, `mdp/`, `models/` | `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`; rewards/terminations **reflected** from `env_cfg`; `rl_cfg` holds no task knowledge; `strict=True` refuses to start on unevaluable terms | **ours** |
| Proof the objective is unchanged | not reported | `tests/test_reward_equivalence.py`: every term, the observation frame and the termination **equal the legacy implementation**, worst `3.7e-9`, exit code gated | **ours** |
| Non-analytic reward terms | implemented directly per task; no learned channel | learned heads supervised by the environment's own per-term values, gated at `corr ≥ 0.9`, with the measured correlation of every term published (§2.4) | **ours** |
| Normalisation | reported as a key ablation factor (single-run comparison) | A/B with a full per-iteration curve: `nse1` 2.4x better at 3k, return +2–6 % (§2.3) | ~ |
| Ablations | configuration table (normalisation, orientation anchor, LR schedule) as described by the project | normalisation A/B + reward-channel study; no H / critic / anchor ablation | **theirs** |
| Velocity controller | full subsystem (`velocity_cfg`, `mdp/velocity.py`, `velocity_commands`, `models/velocity_policy.py`, `rl/velocity_*`, `train/evaluate/play_velocity`, 4 test files, 16 KB design doc) — its documentation states the pipeline is validated but **the converged policy and the 672-condition evaluation were not run** | `Policy.encoders["velocity"]` + 930-dim engine-layout history + 672-condition step test; **four configurations measured, all 0/64**, plus the measured ceiling (action sensitivity 11.6 %) | ~ (theirs has the more complete subsystem; both lack a result) |
| Released weights | **yes** — `policy_best.pt` (52.5 MB) + `world_model_best.pt` (35.3 MB) + a 10k preview | no | **theirs** |
| Documentation | 6+ guides (setup, training & playback, LAFAN conversion, foundation, world model, online), English and Chinese | `README.md` / `README_zh.md`, `docs/lessons.md` (403 lines), `docs/RESULTS.md` (213 lines) | **theirs** on onboarding, **ours** on failure analysis |
| Tests | regression tests across data, world model, checkpointing, playback, normalisation, velocity | 1 equivalence test (the contract proof) — much narrower | **theirs** |

**Summary.** `supermimic_mjlab` is the better *paper reproduction*: a 40/40 result on a standard
benchmark in four hours, released weights, a configuration ablation and a complete velocity subsystem
that is documented rather than hand-waved. This repository is the better *platform*: tasks are declared
through a reflected contract whose equivalence is a test, the library is five times larger and measured
for generalisation rather than memorisation, and every negative result ships with the measurement that
produced it. If the goal is to reproduce the paper, use theirs; if the goal is to add a task and trust the
number, use this one.

## 8. Limitations

* **Fall/get-up remains the weak class** (0.6703, 6/6 failures). The paper closes it with 9.8 h on two
  GPUs; not attempted here.
* **Velocity tracking does not converge** (§6). The implementation exists and is exercised by tests; the
  result is a plateau at 6–10 % of commands, with the diagnosis above.
* **BONES-SEED was not used**; the whole-library run uses 22,246 retargeted clips, so failure counts are
  not directly comparable to the paper's 129,514-clip benchmark.
* Rewards for contact-dependent terms cannot be evaluated in imagination; they are dropped (§2.4) rather
  than proxied.

---

## 9. License and acknowledgements

Code MIT (pending the repository owner's choice). Method from Liao et al., *SuperMimic*, CoRL 2026.
mjlab, MuJoCo and MuJoCo Warp are used unmodified. Motion data (LAFAN1 retargets, the internal clip
library) is not redistributed here.
