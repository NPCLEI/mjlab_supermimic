# SuperMimic 通用训练框架（像 PPO 一样调用）

把 SuperMimic 从"一条任务的脚本"抽成 mjlab 里的**通用 runner**：写一个适配器，就能用同一套算法训练任意任务。

## 用法（PPO 风格）

```python
from src.tasks.supermimic import SuperMimicCfg, SuperMimicRunner
from src.tasks.supermimic.adapters.tracking import MotionTrackingAdapter

cfg = SuperMimicCfg(run_name="my_run", log_root="logs")
cfg.env.num_envs = 2048
cfg.schedule.max_hours = 3.0

env     = MotionTrackingAdapter.build_env(cfg, npz_dir="motion_lib_300")
adapter = MotionTrackingAdapter(env)

runner = SuperMimicRunner(env, cfg, "logs/my_run", "cuda:0", adapter=adapter)
runner.learn()                 # 训练（= PPO 的 runner.learn）
runner.save("ckpt_latest.pt")  # 保存
```

命令行等价形式：

```bash
PYTHONPATH=. python -m src.tasks.supermimic.train \
  --adapter tracking --data.npz-dir motion_lib_300 \
  --env.num-envs 2048 --schedule.max-hours 3.0 \
  --log-root logs --run-name demo
```

## 结构

```
src/tasks/supermimic/
  engine/          # 参考实现逐字节 vendored（世界模型/想象/策略/GAE/回放缓冲区）
  rl/
    adapter.py     # TaskAdapter 协议（唯一要为新任务实现的东西）
    config.py      # SuperMimicCfg = 参考实现的 TrainConfig
    runner.py      # SuperMimicRunner：收集 -> 拟合世界模型 -> 想象中改进策略
  adapters/
    tracking.py    # 参考适配器：任意 npz 动作库的 G1 tracking
    mjlab_task.py  # 任意 mjlab MotionCommand 任务的桥接（含 goal 扩展钩子）
  train.py         # CLI（等价于 PPO 的 train.py）
  smoke.py         # 端到端冒烟测试
```

算法层（`engine/`）不依赖任何具体任务；任务差异全部收敛到 `TaskAdapter`：

| 方法 | 作用 |
|---|---|
| `build_lib()` | 参考库：`ref`/`bodies`/`goal_window`/`last_frame_of` |
| `build_heads(env)` | 在**预测状态**上算解析奖励/终止/FK，并提供 `reward_cfg.max_step_reward` |
| `state(env)` | 当前 71 维模拟状态 `qpos｜qvel` |
| `observation(env)` | `{"policy": (N, obs), "goal": (N, goal)}` |
| `env_action(a)` / `policy_action(a)` | 策略输出与动作空间映射（默认恒等） |
| `xi(env)` | 模拟上下文（ξ，33 维；无域随机化时常量） |
| `frame_and_clip(env)` | 参考帧/片段索引（无参考动作的任务可省） |
| `buffer(env)` | 回放缓冲区（默认用 runner 内置的） |
| `on_reset(env, ids)` | 复位后的观测/动作历史维护 |

## 新任务要写多少

以 `adapters/mjlab_task.py` 为模板：继承式的 `MjlabTrackingAdapter` 已经实现了
mjlab `MotionCommand` 任务的 `state`/`observation`/`frame_and_clip`/`lib`，新任务通常只需覆盖：

```python
class MyAdapter(MjlabTrackingAdapter):
    def observation(self, env):
        # 例如 mixed_cmd 的 token goal：
        return {"policy": proprio(env), "goal": tokens(env)}

    def build_heads(self, env):
        # 用任务自己的 8 项解析奖励，而不是 tracking 的那 9 项
        ...
```

## 已验证

* `smoke.py`：单条动作、128 envs、400 迭代，世界模型 1 步误差从 ~0.25 降到 <0.05，
  训练回报为正，checkpoint 可保存 → 断言通过。
* 同一引擎 + `MotionTrackingAdapter` 在 300 条动作上训练 3 h，Appendix-F 真实成绩
  **0.716（12/300 失败）**，世界模型 8 步误差 **0.0081**。

## 与参考实现的两处环境层差异（非算法）

1. `envs/tracking_env_cfg.py` 的接触栈容量 `nconmax 35→200`、`njmax 250→1024`：
   否则部分动作触发 `nefc overflow`，物理出 NaN 并污染回放缓冲区。
2. `learn()` 开头显式 `env.reset()`：mjlab 的数据缓冲区在首次 reset 前是零，
   零四元数写进回放缓冲区会让世界模型前向 `quat_normalize(0)` 产生 NaN。
