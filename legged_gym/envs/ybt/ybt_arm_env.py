"""Armed YBT: YBTRobot with the six arm joints driven as a frozen payload.

Nothing else about the environment changes -- the arm's DOFs are still observed and
still consume six policy outputs, they are just multiplied by an action scale of
zero, which turns their PD law into a hold at `default_joint_angles`.  See
ybt_arm_config.py for the pose that hold pins the arm to and why it was chosen.

The one physical property this class does add is rotor inertia
(`control.arm_armature`) on the arm's DOFs, applied per joint while the asset is
being created.  Without it `arm_joint_6` has a mass-matrix diagonal of 0 and *no*
position gain is stable at the 200 Hz physics step, so the tool roll rings at
~4 rad/s forever.  The legs are deliberately left at the armless task's armature of
0, so their dynamics are unchanged.
"""
from legged_gym.envs.ybt.ybt_env import YBTRobot


class YBTArmRobot(YBTRobot):
    ARM_JOINT_PREFIX = "arm_joint"
    NUM_ARM_JOINTS = 6

    def _arm_dof_indices(self):
        """The six arm DOF indices, found by name.

        Isaac Gym reports DOFs in URDF declaration order, which is not something to
        depend on; looking the arm up by name keeps that out of the picture.
        """
        arm_idx = [i for i, name in enumerate(self.dof_names)
                   if name.startswith(self.ARM_JOINT_PREFIX)]
        if len(arm_idx) != self.NUM_ARM_JOINTS:
            raise ValueError("ybt_arm expects {} arm joints, found {} in {}".format(
                self.NUM_ARM_JOINTS, len(arm_idx), self.dof_names))
        return arm_idx

    def _process_dof_props(self, props, env_id):
        # runs inside _create_envs, i.e. before _init_buffers, and what comes back
        # from here is what reaches set_actor_dof_properties
        props = super()._process_dof_props(props, env_id)
        props["armature"][self._arm_dof_indices()] = self.cfg.control.arm_armature
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.action_scale[self._arm_dof_indices()] = self.cfg.control.arm_action_scale
