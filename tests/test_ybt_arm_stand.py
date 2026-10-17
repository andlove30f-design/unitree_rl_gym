"""Config, scripted-trajectory and curriculum tests for the standing armed-YBT task.

No simulator: everything worth testing here -- the scripted reference, the torque bias,
the reset, the drift termination and the curriculum -- is plain tensor arithmetic on the
environment object, so the object is built with `object.__new__` and only the buffers
the code under test reads are filled in, the way tests/test_ybt.py already does it.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import isaacgym  # must precede torch
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfg
from legged_gym.envs.ybt.ybt_arm_env import YBTArmRobot
from legged_gym.envs.ybt.ybt_arm_stand_config import (YBTFlatArmStandCfg,
                                                      YBTFlatArmStandCfgPPO)
from legged_gym.envs.ybt.ybt_arm_stand_env import YBTArmStandRobot
from legged_gym.envs.ybt.ybt_env import validate_ybt_asset

ARM = ["arm_joint_{}".format(i) for i in range(1, 7)]
# The twelve leg joints, by their real names: default_dof_pos is built by looking
# `dof_names` up in the config, so the fake order has to use names the config knows.
# Arm last, so `arm_idx` is 12..17 the way slicing here assumes.
LEGS = sorted(name for name in YBTFlatArmStandCfg().init_state.default_joint_angles
              if not name.startswith("arm_joint"))
SIM_DT = 0.005
ARM_LIMIT = 50.0   # stands in for the URDF efforts; only the clip test looks at it


def make_robot(num_envs=4, cfg=None):
    """A YBTArmStandRobot with its buffers built, but no PhysX.

    `_init_buffers` is exercised for real -- only its super() call is replaced with the
    part of LeggedRobot's that the arm code depends on -- so every test below also
    covers the buffer construction.
    """
    cfg = cfg if cfg is not None else YBTFlatArmStandCfg()
    robot = object.__new__(YBTArmStandRobot)
    robot.cfg = cfg
    robot.device = "cpu"
    robot.num_envs = num_envs
    robot.num_dofs = robot.num_dof = robot.num_actions = len(LEGS) + len(ARM)
    robot.dof_names = LEGS + ARM
    robot.sim_params = SimpleNamespace(dt=SIM_DT)
    robot.dt = cfg.control.decimation * SIM_DT
    robot.max_episode_length = math.ceil(cfg.env.episode_length_s / robot.dt)
    robot.dof_state = torch.zeros(num_envs, robot.num_dof, 2)
    robot.dof_pos = robot.dof_state[..., 0]
    robot.dof_vel = robot.dof_state[..., 1]
    robot.torque_limits = torch.full((robot.num_dof,), ARM_LIMIT)
    hard = torch.stack((torch.full((robot.num_dof,), -4.0),
                        torch.full((robot.num_dof,), 4.0)), dim=1)
    robot.dof_pos_limits = hard
    robot.gym, robot.sim = Mock(), object()

    def base_buffers(env):
        # stands in for LeggedRobot._init_buffers + YBTArmRobot._init_buffers
        env.p_gains = torch.arange(1.0, len(LEGS) + len(ARM) + 1.0) * 10.0
        env.d_gains = torch.full((env.num_dof,), 2.0)
        env.action_scale = torch.full((env.num_dof,), float(cfg.control.action_scale))
        env.action_scale[env._arm_dof_indices()] = cfg.control.arm_action_scale
        env.default_dof_pos = torch.tensor([[
            float(cfg.init_state.default_joint_angles[name]) for name in env.dof_names]])

    with patch.object(YBTArmRobot, "_init_buffers", base_buffers):
        robot._init_buffers()
    return robot


def arm_pose(cfg, pose):
    """The reference angles as a plain vector, built from the config alone."""
    return torch.tensor([float(pose[name]) for name in ARM])


def all_scales(cfg):
    """Every reward scale the config answers to, including the inherited ones.

    `vars()` on the scales class only shows what that one class declares, and the
    standing task declares exactly one entry.
    """
    merged = {}
    for klass in reversed(type(cfg.rewards.scales).__mro__):
        merged.update(vars(klass))
    return {name: value for name, value in merged.items() if not name.startswith("_")}


class ArmStandTests(unittest.TestCase):
    def setUp(self):
        self.cfg = YBTFlatArmStandCfg()
        self.path = validate_ybt_asset(self.cfg)

    # --- registration and configuration ---------------------------------------
    def test_registration(self):
        self.assertIn("ybt_arm_stand", task_registry.task_classes)
        self.assertIs(task_registry.get_task_class("ybt_arm_stand"), YBTArmStandRobot)
        self.assertTrue(issubclass(YBTArmStandRobot, YBTArmRobot))
        env_cfg, train_cfg = task_registry.get_cfgs("ybt_arm_stand")
        # the observation width is what makes a ybt_arm checkpoint loadable at all
        self.assertEqual((env_cfg.env.num_actions, env_cfg.env.num_observations), (18, 66))
        self.assertEqual(train_cfg.runner.experiment_name, "flat_ybt_arm_stand")
        self.assertFalse(train_cfg.runner.resume)

    def test_standing_commands(self):
        # Every command pinned to zero turns the two tracking rewards into "do not move"
        # terms.  heading_command stays on: with a zero target it is a free yaw hold.
        self.assertTrue(self.cfg.commands.heading_command)
        self.assertFalse(self.cfg.commands.curriculum)
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"):
            self.assertEqual(getattr(self.cfg.commands.ranges, name), [0.0, 0.0], name)

    def test_only_the_intended_reward_changes(self):
        # Redeclaring the whole `rewards` block instead of the one scale is the easy way
        # to silently drop the six terms the armless task overrides.
        inherited_cfg = YBTFlatArmCfg()
        scales = all_scales(inherited_cfg)
        for name, value in scales.items():
            if name == "feet_clearance":
                continue
            self.assertEqual(getattr(self.cfg.rewards.scales, name), value, name)
        # the clearance term pays for lifting a foot to 0.05..0.09 m, which is the
        # opposite of standing still
        self.assertGreater(scales["feet_clearance"], 0.0)
        self.assertEqual(self.cfg.rewards.scales.feet_clearance, 0.0)
        # still enough terrain override left to be flat, still no shoving
        self.assertEqual(self.cfg.terrain.mesh_type, inherited_cfg.terrain.mesh_type)
        self.assertFalse(self.cfg.domain_rand.push_robots)
        # the drift limit has to stay well inside the slot the env is placed in
        self.assertLess(self.cfg.stand.max_drift, self.cfg.env.env_spacing)

    def test_the_arm_gains_are_retuned_and_the_legs_are_not(self):
        # The arm's damping is what makes it track instead of ring (see the config's
        # docstring), and it applies to this task only -- a resumed walking checkpoint
        # has to meet exactly the stance it was trained in.
        walking = YBTFlatArmCfg.control
        for name in ARM:
            self.assertAlmostEqual(self.cfg.control.stiffness[name],
                                   2.0 * walking.stiffness[name], places=6, msg=name)
            self.assertAlmostEqual(self.cfg.control.damping[name],
                                   0.25 * walking.damping[name], places=6, msg=name)
            self.assertNotEqual(self.cfg.control.stiffness[name], walking.stiffness[name])
        # and the legs are bit-identical, which is what protects a warm start.  The leg
        # entries are keyed by suffix ("hip_joint"), not by the joint's full name -- the
        # base class matches them against `dof_names` -- so compare keys, not joints.
        legs = [key for key in walking.stiffness if not key.startswith("arm_joint")]
        self.assertTrue(legs)
        for name in legs:
            self.assertEqual(self.cfg.control.stiffness[name], walking.stiffness[name], name)
            self.assertEqual(self.cfg.control.damping[name], walking.damping[name], name)
        # the swing the arm has to follow is the reason the stiffness moved at all
        self.assertEqual(self.cfg.control.arm_action_scale, 0.0)

    def test_every_nonzero_reward_scale_has_a_function(self):
        # _prepare_reward_function getattr()s every non-zero scale at construction time,
        # so a typo is an AttributeError in make_env rather than a silent zero.
        for name, scale in all_scales(self.cfg).items():
            if name == "termination" or not isinstance(scale, (int, float)) or scale == 0:
                continue
            self.assertTrue(hasattr(YBTArmStandRobot, "_reward_" + name), name)

    def test_prepare_reward_function_rejects_missing_curriculum_keys(self):
        # The curriculum reads these two out of episode_sums, and episode_sums only holds
        # the scales that survived the zero-scale pop, so a zero here is a silent no-op
        # curriculum unless something says so.
        robot = make_robot()
        for present in ({"tracking_lin_vel", "orientation"}, {"tracking_lin_vel"},
                        {"orientation"}, set()):
            robot.episode_sums = {key: torch.zeros(2) for key in present}
            with patch.object(LeggedRobot, "_prepare_reward_function", lambda self: None):
                if "orientation" in present and "tracking_lin_vel" in present:
                    robot._prepare_reward_function()
                else:
                    with self.assertRaisesRegex(ValueError, "non-zero scales"):
                        robot._prepare_reward_function()

    # --- the scripted motion --------------------------------------------------
    def test_operation_pose_is_inside_the_joint_limits(self):
        pose = self.cfg.arm_trajectory.operation_pose
        self.assertEqual(set(pose), set(ARM))
        joints = {j.attrib["name"]: j for j in ET.parse(str(self.path)).getroot().findall("joint")}
        for name in ARM:
            limit = joints[name].find("limit").attrib
            lower, upper = float(limit["lower"]), float(limit["upper"])
            # a margin, not just "inside": the PD cannot hold a joint that has to push
            # against its own stop to get there
            self.assertGreater(pose[name] - lower, 0.05, name)
            self.assertGreater(upper - pose[name], 0.05, name)

    def test_folded_pose_is_unchanged(self):
        # The arm's home pose is what the observation block is measured against and what
        # a ybt_arm checkpoint was trained with, so the standing task must not move it.
        inherited = YBTFlatArmCfg().init_state.default_joint_angles
        for name in ARM:
            self.assertEqual(self.cfg.init_state.default_joint_angles[name], inherited[name], name)

    def test_swing_is_wide_enough_to_be_a_disturbance(self):
        # folded_pose.py --sweep measures the arm's centre of mass travelling 0.35 m
        # along this pair of angles, which moves the trunk's by 2.7 cm; growing the
        # amplitude around the folded pose instead only reaches 0.14 m and would teach
        # the legs nothing.  These two angles are what buy that distance.
        home = arm_pose(self.cfg, self.cfg.init_state.default_joint_angles)
        working = arm_pose(self.cfg, self.cfg.arm_trajectory.operation_pose)
        swing = (working - home).abs()
        self.assertGreater(swing[1].item(), 1.5)   # arm_joint_2 leans out over the front
        self.assertGreater(swing[2].item(), 1.5)   # arm_joint_3 unfolds the elbow

    def test_target_is_home_at_scale_zero(self):
        robot = make_robot()
        robot.arm_phase = torch.linspace(0.0, 6.0, robot.num_envs)
        robot.arm_scale = 0.0
        robot._refresh_arm_target()
        self.assertTrue(torch.equal(robot.arm_target,
                                    robot.arm_home.unsqueeze(0).expand(robot.num_envs, len(ARM))))
        # and with the phase at "fully extended" it is still home, i.e. the phase and the
        # amplitude are independent
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot._refresh_arm_target()
        self.assertTrue(torch.equal(robot.arm_target[:, 1],
                                    robot.arm_home[1].expand(robot.num_envs)))

    def test_target_reaches_the_working_pose_at_full_amplitude(self):
        robot = make_robot()
        working = arm_pose(self.cfg, self.cfg.arm_trajectory.operation_pose)
        robot.arm_scale = 1.0
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot._refresh_arm_target()
        self.assertTrue(torch.allclose(robot.arm_target,
                                       working.unsqueeze(0).expand(robot.num_envs, len(ARM)),
                                       atol=1e-5))
        robot.arm_phase = torch.zeros(robot.num_envs)
        robot._refresh_arm_target()
        self.assertTrue(torch.allclose(robot.arm_target,
                                       robot.arm_home.unsqueeze(0).expand(robot.num_envs, len(ARM)),
                                       atol=1e-6))

    def test_reference_drives_the_arm_columns_and_only_them(self):
        robot = make_robot()
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot.arm_scale = 1.0
        robot._refresh_arm_target()
        actions = torch.rand(robot.num_envs, robot.num_actions) * 2.0 - 1.0
        out = robot._compute_torques(actions)
        base = LeggedRobot._compute_torques(robot, actions)
        # legs must come out bit for bit identical, or this task quietly changes the
        # walking dynamics it inherits
        self.assertTrue(torch.equal(out[:, :len(LEGS)], base[:, :len(LEGS)]))
        expected = torch.clip(
            robot.arm_p_gains * (robot.arm_target - robot.dof_pos[:, robot.arm_idx])
            - robot.arm_d_gains * robot.dof_vel[:, robot.arm_idx],
            -robot.arm_limits, robot.arm_limits)
        self.assertTrue(torch.equal(out[:, robot.arm_idx], expected))
        # what the base class alone would ask for is a saturated hold at the folded pose
        self.assertEqual(base[:, robot.arm_idx].abs().max().item(), ARM_LIMIT)

    def test_arm_holding_its_reference_is_commanded_nothing(self):
        # The additive form (clip(base) + p*(reference - home)) failed exactly here: with
        # the arm sitting on its reference the base hold is already saturated the wrong
        # way, so the sum asked for full effort instead of nothing and the joint ran to
        # its stop.
        robot = make_robot()
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot.arm_scale = 1.0
        robot._refresh_arm_target()
        robot.dof_pos[:, robot.arm_idx] = robot.arm_target
        robot.dof_vel[:, robot.arm_idx] = 0.0
        out = robot._compute_torques(torch.zeros(robot.num_envs, robot.num_actions))
        self.assertTrue(torch.all(out[:, robot.arm_idx] == 0.0))
        # and the reference really is away from home, or the test above proves nothing
        self.assertGreater((robot.arm_target[0] - robot.arm_home).abs().max().item(), 1.0)

    def test_arm_torque_respects_the_effort_limit(self):
        robot = make_robot()
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot.arm_scale = 1.0
        robot._refresh_arm_target()
        robot.dof_pos[:, robot.arm_idx] = robot.arm_home   # 2.1 rad from the reference
        out = robot._compute_torques(torch.zeros(robot.num_envs, robot.num_actions))
        self.assertTrue(torch.all(out[:, robot.arm_idx].abs() <= robot.arm_limits + 1e-5))
        # shoulder pulling the arm *down* toward the working pose, not up off its stop
        self.assertEqual(out[0, robot.arm_idx[1]].item(), -ARM_LIMIT)

    def test_zero_amplitude_is_the_frozen_arm_task(self):
        robot = make_robot()
        robot.arm_scale = 0.0
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot._refresh_arm_target()
        actions = torch.rand(robot.num_envs, robot.num_actions) * 2.0 - 1.0
        self.assertTrue(torch.equal(robot._compute_torques(actions),
                                    LeggedRobot._compute_torques(robot, actions)))

    def test_step_advances_the_phase_once_per_policy_step(self):
        robot = make_robot()
        expected = 2.0 * math.pi * self.cfg.arm_trajectory.frequency * robot.dt
        robot.arm_phase = torch.zeros(robot.num_envs)
        with patch.object(LeggedRobot, "step", autospec=True) as base_step:
            robot.step(torch.zeros(robot.num_envs, robot.num_actions))
        base_step.assert_called_once()
        self.assertTrue(torch.allclose(robot.arm_phase,
                                       torch.full((robot.num_envs,), expected), atol=1e-6))
        # one 20 ms staircase runs the length of the 4 physics substeps, so the torque
        # path must not touch the phase
        before = robot.arm_phase.clone()
        for _ in range(self.cfg.control.decimation):
            robot._compute_torques(torch.zeros(robot.num_envs, robot.num_actions))
        self.assertTrue(torch.equal(robot.arm_phase, before))

    def test_robot_phase_starts_spread_out(self):
        robot = make_robot(num_envs=512)
        self.assertGreater(robot.arm_phase.std().item(), 1.0)
        self.assertTrue(torch.all(robot.arm_phase >= 0.0))
        self.assertTrue(torch.all(robot.arm_phase < 2.0 * math.pi))

    # --- resets and termination ----------------------------------------------
    def test_reset_puts_the_arm_on_its_reference(self):
        robot = make_robot()
        robot.arm_scale = 1.0
        robot.arm_phase = torch.full((robot.num_envs,), math.pi / 2.0)
        robot._refresh_arm_target()
        robot.dof_vel[:] = 7.0
        ids = torch.tensor([1, 3])
        noise = self.cfg.init_state.joint_angle_noise
        with patch("legged_gym.envs.ybt.ybt_env.gymtorch.unwrap_tensor",
                   side_effect=lambda tensor: tensor):
            robot._reset_dofs(ids)
        self.assertTrue(torch.equal(robot.dof_pos[ids][:, robot.arm_idx],
                                    robot.arm_target[ids]))
        self.assertTrue(torch.all(robot.dof_vel[ids] == 0.0))
        legs = torch.abs(robot.dof_pos[ids][:, :len(LEGS)]
                         - robot.default_dof_pos[:, :len(LEGS)])
        self.assertLessEqual(legs.max().item(), noise + 1e-6)
        # environments that were not reset are untouched, including their arm
        self.assertTrue(torch.all(robot.dof_vel[[0, 2]] == 7.0))
        self.assertTrue(torch.all(robot.dof_pos[[0, 2]][:, robot.arm_idx] == 0.0))

    def test_termination_catches_a_wandering_robot(self):
        robot = make_robot()
        robot.reset_buf = torch.zeros(robot.num_envs, dtype=torch.bool)
        robot.base_pos = torch.zeros(robot.num_envs, 3)
        robot.env_origins = torch.zeros(robot.num_envs, 3)
        limit = self.cfg.stand.max_drift
        robot.base_pos[1, 1] = limit - 0.01     # inside its slot
        robot.base_pos[2, 0] = limit + 0.01     # wandered out of it
        with patch.object(LeggedRobot, "check_termination", lambda self: None):
            robot.check_termination()
        self.assertFalse(robot.reset_buf[1].item())
        self.assertTrue(robot.reset_buf[2].item())

    # --- the curriculum -------------------------------------------------------
    def test_arm_statistics_are_per_second_and_skip_unstepped_envs(self):
        robot = make_robot()
        robot.dt = 0.02
        robot._reset_arm_stat()
        lengths = torch.tensor([0, 500, 1000, 10])       # env 0 was just reset
        robot.episode_length_buf = lengths.clone()
        per_step = {"tracking_lin_vel": 0.5, "orientation": -0.02}
        robot.episode_sums = {
            key: torch.tensor([value * n * robot.dt for n in lengths])
            for key, value in per_step.items()}
        robot._collect_arm_stat(torch.arange(robot.num_envs))
        self.assertEqual(int(robot._arm_stat["episodes"]), 3)
        self.assertEqual(robot._arm_stat["lengths"].item(), 500 + 1000 + 10)
        for key, value in per_step.items():
            self.assertAlmostEqual(robot._arm_stat[key].item(), 3 * value, places=5)

    def prepare_curriculum(self, episodes=256, length=1000, track=1.0, tilt=0.0):
        robot = make_robot()
        robot.dt = 0.02
        robot.common_step_counter = 0
        robot._arm_reviewed = -10 ** 9
        robot.arm_scale = 0.0
        robot._reset_arm_stat()
        robot._arm_stat["episodes"] += episodes
        robot._arm_stat["lengths"] += episodes * length
        robot._arm_stat["tracking_lin_vel"] += episodes * track
        robot._arm_stat["orientation"] += episodes * tilt
        return robot

    def test_curriculum_grows_on_a_healthy_stand(self):
        robot = self.prepare_curriculum()
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale, self.cfg.arm_trajectory.curriculum_step)
        # judged statistics are consumed, so the next review does not see them twice
        self.assertEqual(robot._arm_stat["episodes"].item(), 0.0)

    def test_curriculum_does_not_outrun_its_interval(self):
        robot = self.prepare_curriculum()
        robot._update_arm_curriculum()
        robot._arm_stat["episodes"] += 256     # a fresh, equally good batch
        robot._arm_stat["lengths"] += 256 * 1000
        robot._arm_stat["tracking_lin_vel"] += 256.0
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale, self.cfg.arm_trajectory.curriculum_step)
        robot.common_step_counter += int(self.cfg.arm_trajectory.curriculum_interval_s / robot.dt)
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale, 2 * self.cfg.arm_trajectory.curriculum_step)

    def test_curriculum_waits_for_enough_episodes(self):
        robot = self.prepare_curriculum(episodes=self.cfg.arm_trajectory.curriculum_min_episodes - 1)
        robot._update_arm_curriculum()
        self.assertEqual(robot.arm_scale, 0.0)
        # nothing is judged, so nothing is consumed either
        self.assertEqual(int(robot._arm_stat["episodes"]),
                         self.cfg.arm_trajectory.curriculum_min_episodes - 1)

    def test_curriculum_holds_on_each_failing_gate(self):
        trajectory = self.cfg.arm_trajectory
        cases = {
            # 500 of a 1000-step episode is well under the 0.9 fraction it asks for
            "episode ended early": {"length": 500.0},
            "did not stay put": {"track": trajectory.min_tracking_reward - 0.05},
            # orientation counts down, so a *more negative* average means more tilt
            "leaned over": {"tilt": -(trajectory.max_tilt_cost + 0.01)},
        }
        for label, overrides in cases.items():
            with self.subTest(label):
                robot = self.prepare_curriculum(**overrides)
                robot._update_arm_curriculum()
                self.assertEqual(robot.arm_scale, 0.0, label)

    def test_curriculum_clamps_at_the_ceiling(self):
        trajectory = self.cfg.arm_trajectory
        robot = self.prepare_curriculum()
        robot.arm_scale = trajectory.curriculum_max - trajectory.curriculum_step / 2.0
        robot._update_arm_curriculum()
        self.assertEqual(robot.arm_scale, trajectory.curriculum_max)
        # and once there it stays, however well the robot stands
        robot = self.prepare_curriculum()
        robot.arm_scale = trajectory.curriculum_max
        robot._update_arm_curriculum()
        self.assertEqual(robot.arm_scale, trajectory.curriculum_max)

    def test_set_arm_scale_repins_the_reference(self):
        robot = make_robot()
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot.set_arm_scale(self.cfg.arm_trajectory.curriculum_max)
        self.assertTrue(torch.allclose(robot.arm_target,
                                       arm_pose(self.cfg, self.cfg.arm_trajectory.operation_pose)
                                       .unsqueeze(0).expand(robot.num_envs, len(ARM)), atol=1e-5))
        robot.set_arm_scale(0.0)
        self.assertTrue(torch.equal(robot.arm_target,
                                    robot.arm_home.unsqueeze(0).expand(robot.num_envs, len(ARM))))


if __name__ == "__main__":
    unittest.main()
