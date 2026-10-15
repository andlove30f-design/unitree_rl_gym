"""Training configuration for the armed YBT (quadruped + RM65-B arm).

First goal is walking, not manipulation, so the arm is **frozen** and only acts as
7.173 kg of payload on the trunk.  Everything else is inherited verbatim from
YBTFlatCfg -- same rewards, same commands, same terrain, same leg gains -- so this
task differs from the armless `ybt` task in as few ways as possible.

How the freeze works
--------------------
LeggedRobot._compute_torques scales the policy action by `self.action_scale`, a
per-joint vector seeded from `control.action_scale` and overridden per joint by the
task (see YBTArmRobot._init_buffers).  The six arm joints get 0.0, so their torque
collapses to

    p_gains * (default_dof_pos - dof_pos) - d_gains * dof_vel

i.e. a PD hold at `default_joint_angles`, with the policy's output ignored.  To
unfreeze the arm later, set `arm_action_scale` to the leg value (0.25) and give the
arm joints real reward terms; the rest of the task needs no change.

The folded arm pose
-------------------
`default_joint_angles` doubles as the held pose, so the arm is stowed folded against
the back rather than standing straight up.  The angles below were picked by forward
kinematics over the collision-mesh bounds (see resources/robots/ybt_arm/README.md
for the measurements):

  * lowest point of any arm collision geometry: z = +0.140 m in the trunk frame --
    clear of the top deck (+0.1012) and of the trunk's collision box (+0.09), and
    the hand is also behind that box's -x face, so it cannot push into the body;
  * arm mass centre drops from z = 0.552 (upright) to 0.321, and the arm's second
    moment about the trunk origin from 1.918 to 0.742 kg m^2 -- the point of the
    fold, since the upright pose is already statically balanced;
  * cost of the fold: a constant 7.1 N m at arm_joint_2 (limit 60), well inside what
    the PD below holds the pose with. Re-check any pose change with
    resources/robots/ybt_arm/tools/folded_pose.py; it reads these angles straight out
    of this file.
"""
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg, YBTFlatCfgPPO


class YBTFlatArmCfg(YBTFlatCfg):
    """Flat-ground walking for the 68.84 kg armed YBT, arm frozen as payload."""

    class env(YBTFlatCfg.env):
        num_actions = 18  # 12 legs + 6 arm joints
        num_observations = 12 + 3 * 18  # base env layout: 12 + 3 * num_actions

    class init_state(YBTFlatCfg.init_state):
        # legs first, then the arm; arm_joint_2/3/5 fold the arm backwards over the
        # rear deck (flip all three signs to fold forwards instead)
        default_joint_angles = {
            **YBTFlatCfg.init_state.default_joint_angles,
            "arm_joint_1": 0.0,
            "arm_joint_2": 1.0,
            "arm_joint_3": 2.0,
            "arm_joint_4": 0.0,
            "arm_joint_5": 0.8,
            "arm_joint_6": 0.0,
        }

    class control(YBTFlatCfg.control):
        # The arm's six joints cannot share one gain. Their mass-matrix diagonals,
        # from the URDF, are 0.157, 0.127, 0.088, 0.0039, 0.0019 and 0.0 kg m^2 going
        # out from the base, so a single k=150 sits far above the stability limit for
        # dt=0.005 at the wrist -- it chattered at 11 rad/s with the torque pinned to
        # its 10 N m limit and 0.4 rad of position error -- while being limp at the
        # shoulder. Each pair below instead targets the same closed-loop frequency,
        # w = 55 rad/s with critical damping (k = w^2 M, d = 2 w M), against that
        # joint's own inertia plus `arm_armature`.
        #
        # `arm_armature` is rotor inertia, and it is not optional for the wrists: a
        # joint whose mass-matrix diagonal is 0 has a zero upper bound on k at this
        # step size, so arm_joint_6 rings at 4 rad/s with any gain at all. On a real
        # RM65-B the servo's rotor inertia is what dominates a wrist anyway. It is
        # applied to the arm's six DOFs only (YBTArmRobot._process_dof_props), so the
        # legs keep the armless task's dynamics exactly.
        arm_armature = 0.002
        # Held with these gains the arm drifts 0.042 rad off its pose at rest and its
        # worst joint sits at 12% of its torque limit (arm_joint_2, carrying the folded
        # arm's 7.1 N m over a 0.19 m lever). Right after each reset the landing impact
        # every robot in this task takes throws it a little further -- 0.08 rad, hitting
        # the effort limit for a moment -- and then it settles.
        stiffness = {**YBTFlatCfg.control.stiffness,
                     "arm_joint_1": 480.0, "arm_joint_2": 390.0, "arm_joint_3": 270.0,
                     "arm_joint_4": 18.0, "arm_joint_5": 12.0, "arm_joint_6": 6.0}
        damping = {**YBTFlatCfg.control.damping,
                   "arm_joint_1": 17.5, "arm_joint_2": 14.0, "arm_joint_3": 10.0,
                   "arm_joint_4": 0.65, "arm_joint_5": 0.45, "arm_joint_6": 0.2}
        # 0.0 freezes the arm. Same as the legs once the arm is trained for real.
        arm_action_scale = 0.0

    class asset(YBTFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/ybt_arm/urdf/ybt_arm.urdf"
        name = "ybt_arm"
        # "arm" rather than "arm_link": arm_base_link has to match too. Hitting the
        # ground with the arm is penalised but does not end the episode -- only the
        # trunk does that, as for the armless robot.
        penalize_contacts_on = YBTFlatCfg.asset.penalize_contacts_on + ["arm"]


class YBTFlatArmCfgPPO(YBTFlatCfgPPO):
    class runner(YBTFlatCfgPPO.runner):
        experiment_name = "flat_ybt_arm"
