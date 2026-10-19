"""Walking with the scripted arm: the arm works while the legs track velocity commands.

Everything this task does differently from `ybt_arm_stand` is in the config -- the arm
script and its retuned PD gains are the same, and the legs inherit the walking task's
commands and rewards.  The class body is therefore empty.

It deliberately does not define `check_termination`.  Attribute lookup therefore finds
`LeggedRobot.check_termination` -- trunk contact, roll/pitch limits, timeout -- and the
standing task's drift rule (`cfg.stand.max_drift`) does not apply.  A walking robot is
supposed to leave its origin.
"""
from legged_gym.envs.ybt.ybt_arm_scripted_env import YBTArmScriptedRobot


class YBTArmWalkRobot(YBTArmScriptedRobot):
    """Velocity tracking while the arm runs the scripted working motion."""
