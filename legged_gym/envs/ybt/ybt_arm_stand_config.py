"""Standing balance for the armed YBT: legs hold the trunk still while the arm works.

The second stage of the armed quadruped, after `ybt_arm` (arm frozen as a payload,
robot walks).  Here every velocity command is zero and the legs have one job: keep the
trunk where it is, level, while the arm runs a scripted working motion on its own.
The robot is *standing*, not walking, so nothing pays for lifting a foot -- a small
corrective shuffle is allowed, and the velocity rewards decide whether it helps.

The arm is still not policy-controlled.  Its six outputs are ignored exactly as in
`ybt_arm` (`control.arm_action_scale` is 0.0), but instead of holding one pose its PD
*reference* is driven by a script, in YBTFlatArmStandRobot.  Least surprising knobs:

    control.arm_action_scale      0.0 = script drives the arm, policy cannot
    arm_trajectory.operation_pose the pose the arm works in (see below)
    arm_trajectory.frequency      how fast it works, Hz
    arm_trajectory.curriculum_*   how the difficulty grows
    rewards.scales                inherited from ybt_arm, minus feet_clearance

Why the arm has to actually reach out
-------------------------------------
The arm's moving links are 5.31 kg of the robot's 68.84 kg, so the swing is only felt
through where that mass goes.  The pure-inertial reaction is small -- `I * alpha` is a
few N m against the ~100 N m it takes to tip the whole robot -- while the mass-centre
travel moves the trunk's centre of mass by 0.077x of itself.  So the motion below
travels the arm's centre of mass 0.37 m, which moves the trunk's by 2.9 cm: measured,
parking the arm at each end of that line on the same robots, the front feet pick up 42 N
of the robot's 675 N and the trunk pitches 0.5 degrees.  Growing the amplitude *around
the folded pose* instead, which is the obvious guess, moves only 0.14 m and would teach
the legs nothing.  0.37 m is also close to the ceiling: the arm runs out of reach before
`--sweep` runs out of patience, and swinging the base joint sideways instead *shortens*
the travel while pushing the arm out over the legs (|y| = 0.52 m).  The honest scale of
this task is in that 42 N: it is a real balance load, but a small one, and a stance that
already stands still will absorb it without working.

Those numbers, the clearance over the trunk and the joint-limit margins all come from

    python resources/robots/ybt_arm/tools/folded_pose.py --sweep 0.0 -1.5 0.0 0.0 0.8 0.6

which walks the line between the folded pose and the working pose.  Run it again after
changing `operation_pose` -- it fails with a reason if the swing would hit the trunk,
reach the ground, leave the joint limits or be too weak to be worth training against.

Why the arm's gains are not the walking task's
----------------------------------------------
Asking the arm to *move* exposes what holding one pose hid: with the walking task's
gains the joints ring instead of tracking.  Measured with the arm parked at the working
pose, 32 envs, nothing else asked of the robot -- no contact anywhere, and more armature
makes no difference -- `arm_joint_3` sat at 3.17 rad/s and 86% of its 30 N m effort
limit, `arm_joint_2` and `arm_joint_5` at 1.20 and 2.37 rad/s.  That is an explicitly
integrated damping term past its stability limit: `d * dt / I < 2` at dt = 0.005 s, and
these joints have less effective inertia about them than the URDF's mass-matrix diagonal
that sized the gains.  It also shakes the robot rather than pushing it: the same ring
threw the arm 0.12 rad behind its reference and put a 19 Hz vibration into the trunk.

A quarter of the damping removes it completely -- the same pose then costs `arm_joint_3`
13% of its limit, which is its gravity load and nothing else, and halving the damping
instead (d/2) rings just as badly, so a quarter is on the right side of the line.  Twice
the stiffness then recovers the tracking the swing needs: measured over the full cycle
at 0.5 Hz, the arm follows its reference to 0.014 rad on average and 0.14 rad at worst,
with the hardest joint (`arm_joint_2`) at 49% of its effort limit on average.  Twice is
deliberate rather than more: at three times the stiffness the worst error is 0.12 rad
and `arm_joint_2`'s average duty is 52%, so the extra authority buys very little
accuracy and is worth keeping in reserve.

Only the arm's six pairs are overridden; the legs keep the walking task's gains, and
`ybt_arm` itself is untouched.
"""
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfg, YBTFlatArmCfgPPO


class YBTFlatArmStandCfg(YBTFlatArmCfg):
    """Flat-ground standing balance for the armed YBT, arm on a scripted motion."""

    class commands(YBTFlatArmCfg.commands):
        # Stand still.  Every command is pinned to zero, which turns the two tracking
        # rewards into "do not move" terms for free (they are `exp(-err^2/0.25)` on
        # base velocity).  `heading_command` stays on: with a zero heading target it
        # is a free yaw hold, and the walking task needed it to stop a slow uncorrected
        # yaw bias from walking the robot off in a curve.
        heading_command = True
        curriculum = False

        class ranges(YBTFlatArmCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class control(YBTFlatArmCfg.control):
        # 2x the walking task's arm stiffness and 1/4 of its damping, for the reason in
        # the module docstring.  The six numbers below are
        # `2.0 * YBTFlatArmCfg.control.stiffness[joint]` and
        # `0.25 * YBTFlatArmCfg.control.damping[joint]`, written out because a config
        # class body cannot close over a comprehension of its own attributes.
        #
        # Only the arm's joints appear here.  The legs inherit unchanged, so the walking
        # task's dynamics -- and the policy resumed from it -- are exactly as trained.
        stiffness = {**YBTFlatArmCfg.control.stiffness,
                     "arm_joint_1": 960.0, "arm_joint_2": 780.0, "arm_joint_3": 540.0,
                     "arm_joint_4": 36.0, "arm_joint_5": 24.0, "arm_joint_6": 12.0}
        damping = {**YBTFlatArmCfg.control.damping,
                   "arm_joint_1": 4.375, "arm_joint_2": 3.5, "arm_joint_3": 2.5,
                   "arm_joint_4": 0.1625, "arm_joint_5": 0.1125, "arm_joint_6": 0.05}

    class arm_trajectory:
        """The scripted working motion, interpolated from the folded pose.

        The reference is
            q(t) = folded + scale * (operation_pose - folded) * (1 - cos(2*pi*f*t + phase)) / 2
        so a cycle goes folded -> working pose -> folded, and `scale` (the curriculum)
        shrinks the whole excursion towards the folded pose.  At scale = 0 the arm sits
        exactly where the walking task holds it, which is what makes resuming a
        frozen-arm checkpoint start on-distribution.
        """

        # The pose the arm works in.  Shoulder swings forward (arm_joint_2 from +1.0,
        # which folds it back over the deck, to -1.5, leaning out over the front), the
        # elbow unfolds (joint_3 +2.0 -> 0.0), the wrist settles and the tool rolls,
        # so the motion reads as reaching out and doing something.  joint_1 and
        # joint_4 stay put: they add no horizontal mass-centre travel, only tracking
        # work for the PD and one more thing for the policy to disentangle.
        #
        # -1.5 is where the shoulder stops buying reach: --sweep measures a 0.373 m
        # mass-centre excursion here against 0.373 m at -1.8, so past this the arm
        # only lifts.  joint_2's measured load at the far end is 31% of its limit.
        operation_pose = {
            "arm_joint_1": 0.0,
            "arm_joint_2": -1.5,
            "arm_joint_3": 0.0,
            "arm_joint_4": 0.0,
            "arm_joint_5": 0.8,
            "arm_joint_6": 0.6,
        }
        # 0.5 Hz = one working cycle every 2 s = 10 cycles per episode.  Far below the
        # PD's closed-loop bandwidth (~55 rad/s, see ybt_arm_config.py), so the arm
        # tracks the reference rather than lagging it.  Raising this is the second
        # lever on difficulty (the inertial reaction grows with frequency squared).
        frequency = 0.5

        # Curriculum: `scale` grows while the robot keeps standing.  Grown in
        # reset_idx from statistics accumulated over finished episodes, so the
        # thresholds below are read on the same per-episode averages the training
        # log prints.
        curriculum_start = 0.0
        curriculum_max = 1.0
        curriculum_step = 0.02
        # Do not evaluate on every reset: 12288 envs with 1000-step episodes reset
        # ~12 envs per policy step, so the test would run ~50 times a second and ramp
        # the amplitude to full in under a second of simulated time.
        curriculum_interval_s = 2.0
        curriculum_min_episodes = 256
        # thresholds, all three have to hold to grow one step
        min_episode_fraction = 0.9   # of the full episode length: it did not fall
        min_tracking_reward = 0.8    # per-second tracking_lin_vel: it did not walk off
        # "Per-second orientation cost" is the reward summed over the episode and divided
        # by max_episode_length_s, which is 50x the per-step mean of sum(gravity_xy^2) --
        # so this threshold is sqrt(0.02 / 50) = 0.02 rad = 1.15 deg of RMS trunk tilt,
        # and *not* the 8 deg that taking the square root of the same number suggests.
        # 1.15 deg comes from measurement rather than from that arithmetic: a settled
        # stand reads 0.5 deg, the same stand with the arm at 0.74 of its swing reads
        # 0.9 deg, and the soft stride the task inherits, standing passively with zero
        # actions, sits at 8.7 deg.  So the gate trips when the trunk has lost roughly
        # twice the levelness a healthy stand shows, and it is deliberately not a "did
        # it fall over" bar -- the length gate is that one.
        max_tilt_cost = 0.02

    class stand:
        """Not part of the asset or the policy -- how the task judges a stand."""

        # Terminate when the trunk has wandered this far (m, in XY) from its own slot.
        # `env_spacing` is 3 m, so without a limit a slow drift lets two robots end up
        # in the same volume and PhysX quietly settles them against each other.
        max_drift = 0.6

    class rewards(YBTFlatArmCfg.rewards):
        class scales(YBTFlatArmCfg.rewards.scales):
            # Clearance pays a foot for sitting in a 0.05..0.09 m band.  Standing is
            # the whole objective here and the legs are allowed to shuffle, but they
            # must not be *paid* to lift -- that is how the walking task ended up
            # high-stepping.  Zero, so the reward is never even bound.
            feet_clearance = 0.0

    class domain_rand(YBTFlatArmCfg.domain_rand):
        # The arm is the only disturbance this task wants, and a 0.5 m/s shove every
        # 15 s would swamp it and make the curriculum's survival test noise.
        push_robots = False


class YBTFlatArmStandCfgPPO(YBTFlatArmCfgPPO):
    class runner(YBTFlatArmCfgPPO.runner):
        experiment_name = "flat_ybt_arm_stand"
