"""Standing balance for the armed YBT: legs hold the trunk still while the arm works.

The second stage of the armed quadruped, after `ybt_arm` (arm frozen as a payload,
robot walks).  Here every velocity command is zero and the legs have one job: keep the
trunk where it is, level, while the arm runs a scripted working motion on its own.
The robot is *standing*, not walking, so nothing pays for lifting a foot -- a small
corrective shuffle is allowed, and the velocity rewards decide whether it helps.

The arm's script (`arm_trajectory`) and its retuned PD gains are shared with
`ybt_arm_walk` and live in ybt_arm_scripted_config.py, together with the measurements
behind them; this file only declares what makes the task *standing*.
"""
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfgPPO
from legged_gym.envs.ybt.ybt_arm_scripted_config import YBTFlatArmScriptedCfg


class YBTFlatArmStandCfg(YBTFlatArmScriptedCfg):
    """Flat-ground standing balance for the armed YBT, arm on a scripted motion."""

    class commands(YBTFlatArmScriptedCfg.commands):
        # Stand still.  Every command is pinned to zero, which turns the two tracking
        # rewards into "do not move" terms for free (they are `exp(-err^2/0.25)` on
        # base velocity).  `heading_command` stays on: with a zero heading target it
        # is a free yaw hold, and the walking task needed it to stop a slow uncorrected
        # yaw bias from walking the robot off in a curve.
        heading_command = True
        curriculum = False

        class ranges(YBTFlatArmScriptedCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class stand:
        """Not part of the asset or the policy -- how the task judges a stand."""

        # Terminate when the trunk has wandered this far (m, in XY) from its own slot.
        # `env_spacing` is 3 m, so without a limit a slow drift lets two robots end up
        # in the same volume and PhysX quietly settles them against each other.
        max_drift = 0.6

    class rewards(YBTFlatArmScriptedCfg.rewards):
        class scales(YBTFlatArmScriptedCfg.rewards.scales):
            # Clearance pays a foot for sitting in a 0.05..0.09 m band.  Standing is
            # the whole objective here and the legs are allowed to shuffle, but they
            # must not be *paid* to lift -- that is how the walking task ended up
            # high-stepping.  Zero, so the reward is never even bound.
            feet_clearance = 0.0

    class domain_rand(YBTFlatArmScriptedCfg.domain_rand):
        # The arm is the only disturbance this task wants, and a 0.5 m/s shove every
        # 15 s would swamp it and make the curriculum's survival test noise.
        push_robots = False


class YBTFlatArmStandCfgPPO(YBTFlatArmCfgPPO):
    class runner(YBTFlatArmCfgPPO.runner):
        experiment_name = "flat_ybt_arm_stand"
