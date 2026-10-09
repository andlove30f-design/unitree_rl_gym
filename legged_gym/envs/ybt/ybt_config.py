from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class YBTFlatCfg(LeggedRobotCfg):
    """Starting simulation configuration for the 61.67 kg YBT URDF."""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 48
        num_privileged_obs = None
        episode_length_s = 20.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        curriculum = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.8, 0.8]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]
        # Additive reset noise, NOT the base class's 0.5..1.5 multiplier.
        joint_angle_noise = 0.05  # radians, uniform +/- this value
        default_joint_angles = {
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.6,
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": -0.8,
            "FL_calf_joint": 1.6,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": -0.8,
            "RR_calf_joint": 1.6,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.6,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # Simulation starting gains; do not use these as hardware motor gains.
        stiffness = {"hip_joint": 100.0, "thigh_joint": 150.0, "calf_joint": 150.0}
        damping = {"hip_joint": 3.0, "thigh_joint": 4.0, "calf_joint": 4.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/ybt/urdf/ybt.urdf"
        name = "ybt"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        collapse_fixed_joints = False  # retain all four foot contact bodies
        self_collisions = 1
        flip_visual_attachments = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = False
        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 0.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.50
        max_contact_force = 500.0

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.00001
            orientation = -1.0
            base_height = -1.0
            dof_pos_limits = -10.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz physics; decimation=4 gives 50 Hz policy


class YBTFlatCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "flat_ybt"
        run_name = ""
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        resume = False
