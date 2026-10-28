"""SuperMimic as a reusable mjlab training framework, used like PPO.

The algorithm (world model, imagination, GAE critic, UTD schedule) is the *vendored
reference implementation* kept byte-identical under ``engine/``. What a task supplies is
a single object, :class:`~src.tasks.supermimic.rl.adapter.TaskAdapter`, and then:

.. code-block:: python

    from src.tasks.supermimic import SuperMimicCfg, SuperMimicRunner
    from src.tasks.supermimic.adapters import MotionTrackingAdapter

    env   = MotionTrackingAdapter.build_env(cfg)          # or any ManagerBasedRlEnv
    adpt  = MotionTrackingAdapter(env, cfg)
    runner = SuperMimicRunner(env, cfg, "logs/my_run", "cuda:0", adapter=adpt)
    runner.learn()                                        # <-- like runner.learn() in PPO
    runner.save("ckpt_latest.pt")

Everything task specific lives in the adapter:

======================================  =============================================
``adapter.build_lib()``                 reference library: ``ref``/``bodies``/
                                        ``goal_window``/``last_frame_of``
``adapter.build_heads(env)``            analytic reward / termination / FK on the
                                        *predicted* state
``adapter.observation(env)``            ``{"policy": (N, obs), "goal": (N, goal)}``
``adapter.env_action(a)``               policy output -> env action (identity by default)
``adapter.buffer(env)``                 replay storage (state, a_exec, goal, done)
======================================  =============================================

A new task therefore means: implement those five methods (``adapters/tracking.py`` is
the reference example, ``adapters/mjlab_task.py`` the generic mjlab bridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent / "engine"
if str(_ENGINE) not in sys.path:
  # the vendored package is imported by its original top-level name
  sys.path.insert(0, str(_ENGINE))

from .rl.adapter import TaskAdapter  # noqa: E402
from .rl.config import SuperMimicCfg  # noqa: E402
from .rl.runner import SuperMimicRunner  # noqa: E402

__all__ = ["SuperMimicCfg", "SuperMimicRunner", "TaskAdapter"]
