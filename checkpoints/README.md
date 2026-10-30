# Pretrained weights

| file | contents | size |
|---|---|---|
| `supermimic_library_24k.pt` | policy (8.91 M params), world model (8.82 M), critic, training config, evaluation metadata | 69 MB |

Trained on the **whole library** used in this repository: 22,246 clips / 11.34 M frames / 313 h, retargeted
from [MotionDecode](https://chingmudata.github.io/MotionDecode/) and
[KungfuAthlete](https://arxiv.org/abs/2602.13656). One policy, one world model, no per-clip specialisation.

```python
{
  "policy":     # state dict, obs_dim 930 (10 frames x 93), goal_dim 689 (10 future frames x 68 + 9 anchor)
  "wm":         # world model, s_{t+1} = f(s_t, a_t, xi) over the 71-dim state
  "critic":     # value head
  "config":     # the training config, including the policy section
  "iteration":  24000,
  "meta":       # dataset, evaluation numbers, protocol
}
```

Optimiser states were stripped, so this is an inference artefact; continue training from a full run
checkpoint instead.

## Measured performance

| protocol | normalised return | failures |
|---|---:|---:|
| 300-clip held-out subset, deterministic, one pass per clip from frame 0 (paper Appendix F) | **0.8248** | 8 / 300 |
| the same subset with a per-clip specialist, for scale | 0.7156 | — |
| six fall/get-up clips | 0.6703 | 6 / 6 |
| world model, 1-step normalised error | `nse1` 0.0035, `nse8` 0.0057 | — |

## How to use it

Evaluation (paper Appendix-F protocol, deterministic, no noise, no domain randomisation):

```bash
python scripts/eval_benchmark.py \
  --ckpt checkpoints/supermimic_library_24k.pt \
  --npz-dir /path/to/motion_dir --num-envs 1
```

Interactive playback (reference ghost vs the policy; native viewer or viser in a browser):

```bash
python scripts/play.py \
  --ckpt checkpoints/supermimic_library_24k.pt \
  --npz-dir /path/to/motion_dir --num-envs 4
```

From Python:

```python
from supermimic.eval.benchmark import load_policy_from_checkpoint

policy, cfg = load_policy_from_checkpoint("checkpoints/supermimic_library_24k.pt", "cuda:0")
policy.eval()

# observations the policy expects
#   obs["policy"] : (N, 930)  ten stacked 93-dim frames, term-major
#                   [joint_pos - default (29) | joint_vel (29) | body_ang_vel (3) |
#                    projected_gravity (3) | previous_action (29)]
#   obs["goal"]   : (N, 689)  ten future reference frames (68 each) + 9 anchor dims
#                             (reference anchor pos (3) + rot6d (6) in the robot anchor frame)
action = policy(obs["policy"], obs["goal"])   # (N, 29) joint position targets, clipped to +-1
```

The motion directory must be the same retarget format the training used (one `.npz` per clip with
`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `body_names`), and
the environment must run with anchor observations (`anchor_obs=True`), because `goal_dim` is 689 rather
than 680.

Verified before release: the file loads with `strict=False` reporting no missing or unexpected tensors, the
decoder input is 962 (latent 32 + obs 930), the motion encoder input is 689, and a forward pass returns a
`(2, 29)` action within `[-1, 1]`.

---

## ⚠️ Not recommended for deployment on a real robot

This checkpoint is a **simulation research artefact**. It has been evaluated only inside MuJoCo Warp with the
paper's deterministic protocol, and nothing here has been validated on hardware. Concretely:

* **No sim-to-real work.** No actuator or latency model identified, no system-identification of the real G1,
  no observation delay handling. Domain randomisation was used during data collection but was never checked
  against a physical robot.
* **No state estimation.** Observations assume ideal proprioception (joint positions/velocities, body angular
  velocity, projected gravity) and a reference motion played back from file.
* **No safety layer.** The policy emits joint position targets at 50 Hz with no torque, velocity or
  self-collision guarding, no watchdogs, no fall protection. The `-1..1` clamp is an action-space
  normalisation, not a safety limit.
* **It does fall.** Even on the held-out evaluation subset, 8 of 300 clips fail, and the fall/get-up class
  fails 6 of 6. In simulation that is a number; on hardware it is a machine on the floor.
* **Motion library and reference playback are part of the interface.** The policy expects a reference
  trajectory; it is not a stand-alone controller that can be commanded by velocity or pose.

If hardware work is intended, the minimum path is: identify the actuator and latency model, add a
state-estimation and reference-playback stack, add a supervisor that limits torque/velocity and aborts on
tilt or contact anomalies, and stage the tests on a gantry with an emergency stop — one clip at a time,
starting from the slowest motions in the library.
