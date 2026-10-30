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
# single clip
python -m supermimic.scripts.train \
  --data.npz-dir /path/to/motion_lib_1 \
  --env.num-envs 1024 --env.episode-length-s 20 --env.anchor-obs \
  --schedule.max-hours 8 --run-name one

# whole library (build the pack first)
python scripts/build_library.py --npz-dir /path/to/motions --out data/library_motion
python -m supermimic.scripts.train --data.library data/library_motion --env.num-envs 4096 ...

# evaluate with the paper's Appendix-F protocol, and play it back
python scripts/eval_benchmark.py --ckpt logs/one/ckpt_0018000.pt --npz-dir /path/to/motion_lib_1
python -m supermimic.scripts.play_ref --ckpt logs/one/ckpt_0018000.pt --clip-index 1307
```

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

---

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
