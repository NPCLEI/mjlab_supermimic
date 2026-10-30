# Demo motion: clip 1307

One retargeted G1 motion, published so the released weights can be run without hunting for data.

| | |
|---|---|
| file | `demo/clip_1307/bilibili_ground__Ground__1307.csv.npz` (15 MB) |
| length | **15,792 frames at 50 Hz = 315.84 s** |
| content | a single continuous take: standing, floor work / getting up, walking and turning (pelvis height 0.27–0.82 m) |
| source | our own capture, clip id 1307; released here with the owner's permission |
| results | the published weights score **0.849 normalised return over the full clip, zero failures** on it (`checkpoints/supermimic_library_24k.pt`) |

## Format

One `.npz` per clip, 50 Hz, all quantities in the world frame, `body_names` in the order used by the arrays:

```text
body_names      (14,)            pelvis, knees, ankles, torso, shoulders, elbows, wrists
joint_pos       (T, 29)          joint angles, MuJoCo order
joint_vel       (T, 29)
body_pos_w      (T, 14, 3)       world positions of the 14 tracked bodies
body_quat_w     (T, 14, 4)       wxyz
body_lin_vel_w  (T, 14, 3)
body_ang_vel_w  (T, 14, 3)
```

`body_pos_w[:, 0]` and `body_quat_w[:, 0]` are the pelvis pose, i.e. the free joint; the 29 `joint_pos` values
complete the 36-dim `qpos`. `body_pos_w[:, 7]` is `torso_link`, the anchor body the policy observations and the
anchor-relative goal are expressed against.

Any directory holding one or more of these files works as `--npz-dir`. To use several clips as a training
library, point `--data.npz-dir` at the directory or pack it with `scripts/build_library.py`.

## Try it

```bash
# play the policy against this clip (reference ghost vs. the simulated robot)
python scripts/play.py \
  --ckpt checkpoints/supermimic_library_24k.pt \
  --npz-dir demo/clip_1307 --num-envs 1

# score it with the paper's Appendix-F protocol (deterministic, one pass from frame 0)
python scripts/eval_benchmark.py \
  --ckpt checkpoints/supermimic_library_24k.pt \
  --npz-dir demo/clip_1307 --num-envs 1

# render a grid/still without a viewer
MUJOCO_GL=egl python scripts/render_grid.py \
  --ckpt checkpoints/supermimic_library_24k.pt --npz-dir demo/clip_1307 \
  --clips-file /dev/stdin --out /tmp/clip1307.gif <<< "$(ls "$PWD"/demo/clip_1307/*.npz)"$'\t1307'
```

The environment needs anchor observations (`anchor_obs=True`, which `scripts/play.py` and
`scripts/eval_benchmark.py` set for you), because the released policy was trained with `goal_dim=689`
(680 future-reference dims + 9 anchor dims).

> Reminder from [`../checkpoints/README.md`](../checkpoints/README.md): this is a simulation research
> artefact. Replaying it on a physical robot is not recommended — no actuator/latency model, no state
> estimation, no safety layer, and the policy does fall on some clips.
