"""Task adapters: the only task-specific code needed to train with SuperMimic.

``tracking.py``   the reference adapter: any motion library (npz files) tracked by the
                  G1 with the paper's observation/reward/termination -- this is the
                  setting that reproduces "one policy, hundreds of motions, ~3 h".
``mjlab_task.py`` bridge for an arbitrary mjlab ``ManagerBasedRlEnv`` whose command is a
                  ``MotionCommand``: derives the reference library and the analytic
                  heads from the environment instead of from npz files.
"""

from .tracking import MotionTrackingAdapter

__all__ = ["MotionTrackingAdapter"]
