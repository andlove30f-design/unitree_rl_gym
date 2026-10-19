"""Walking velocity tracking with the scripted arm: the robot works while it goes.

The third stage of the armed quadruped.  It is the `ybt_arm` walking task (random
vx/vy/heading commands, `feet_clearance = 0.3`, `push_robots = True`, base termination)
plus the `ybt_arm_stand` arm script and retuned arm PD gains, both inherited from
YBTFlatArmScriptedCfg.  Nothing else is redeclared: the reward set, the command ranges
and the terrain are the walking task's, so a `ybt_arm` checkpoint is a faithful warm
start and the two experiments stay comparable.

Only the two curriculum gates change, because the legs are now asked to walk rather
than to stand still.  They are looser than the standing task's on purpose -- see the
`arm_trajectory` block below.
"""
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfgPPO
from legged_gym.envs.ybt.ybt_arm_scripted_config import YBTFlatArmScriptedCfg


class YBTFlatArmWalkCfg(YBTFlatArmScriptedCfg):
    """Velocity tracking while the arm runs the scripted working motion."""

    class arm_trajectory(YBTFlatArmScriptedCfg.arm_trajectory):
        # The same script, frequency and ramp schedule as the standing task.  Only the
        # two things the ramp is gated on are re-measured for a walking gait.
        #
        # `min_tracking_reward`: exp(-e^2/0.25) = 0.75 means a combined xy velocity
        # error of 0.268 m/s -- still "following the command", not "surviving while
        # stopped".  The warm-start checkpoint's own training log reads 0.95 with
        # random commands (logs/flat_ybt_arm/Oct15_21-44-18_), so 0.75 leaves room for
        # the dip the swing causes while staying well above a robot that has given up.
        min_tracking_reward = 0.75
        # `max_tilt_cost` is the same quantity as on the standing task -- E[tilt^2] in
        # rad^2, so sqrt(0.08) = 16.2 deg of RMS trunk tilt -- but walking is not level
        # by construction.  Measured on the warm-start checkpoint (ybt_arm model_550,
        # 64 envs, random commands, push_robots on): parked arm 0.0025 (2.9 deg, 1 fall
        # in 3000 steps); full swing 0.0536 (13.3 deg) with 88% of the episodes ending
        # in a fall before their timeout.  The warm start therefore cannot carry the
        # full swing -- which is the point of training this task -- so the gate has to
        # sit *above* that 0.0536 or the ramp would demand a better-than-warm-start
        # policy at the first full-amplitude review.
        #
        # What actually catches a falling robot is `min_episode_fraction`: episodes must
        # run 90% of their 1000 steps.  This gate is the secondary "is it leaning over"
        # check, and erring loose on it is deliberate and asymmetric: too tight leaves
        # `arm_scale` stuck for the whole run -- silent, and the run is wasted -- while
        # too loose only lets the amplitude grow on its fixed schedule (at most
        # `curriculum_step` per `curriculum_interval_s`, so 50 reviews to full).
        max_tilt_cost = 0.08


class YBTFlatArmWalkCfgPPO(YBTFlatArmCfgPPO):
    class runner(YBTFlatArmCfgPPO.runner):
        experiment_name = "flat_ybt_arm_walk"
