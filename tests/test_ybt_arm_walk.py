"""Config and curriculum tests for the walking scripted-arm task.

The scripted-arm machinery itself -- the reference, the torque bias, the reset -- is
shared with `ybt_arm_stand` and covered exhaustively by tests/test_ybt_arm_stand.py.
What this file has to prove is the part that is *not* shared: that the walk task has the
walking task's commands and rewards, that it does not inherit the standing drift rule,
that the two tasks really use one copy of the machinery (not a duplicated one), and that
its curriculum gates are the walking ones.
"""
import math
import unittest
from unittest.mock import patch

import isaacgym  # must precede torch
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfg
from legged_gym.envs.ybt.ybt_arm_env import YBTArmRobot
from legged_gym.envs.ybt.ybt_arm_scripted_config import YBTFlatArmScriptedCfg
from legged_gym.envs.ybt.ybt_arm_scripted_env import YBTArmScriptedRobot
from legged_gym.envs.ybt.ybt_arm_stand_config import YBTFlatArmStandCfg
from legged_gym.envs.ybt.ybt_arm_stand_env import YBTArmStandRobot
from legged_gym.envs.ybt.ybt_arm_walk_config import YBTFlatArmWalkCfg
from legged_gym.envs.ybt.ybt_arm_walk_env import YBTArmWalkRobot
# the no-simulator scaffold and the shared constants, from the sibling test module
from test_ybt_arm_stand import ARM, all_scales, arm_pose, make_robot

# Every method that belongs to "the arm is scripted" rather than to "what the legs
# are asked to do".  Both tasks must resolve all of them to the shared class.
SHARED_METHODS = (
    "_prepare_reward_function", "_init_buffers", "_arm_angle", "_reset_arm_stat",
    "_refresh_arm_target", "set_arm_scale", "step", "_compute_torques", "_reset_dofs",
    "reset_idx", "_collect_arm_stat", "_update_arm_curriculum",
)


class ArmWalkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = YBTFlatArmWalkCfg()

    # --- registration and configuration ---------------------------------------
    def test_registration_and_dimensions(self):
        self.assertIn("ybt_arm_walk", task_registry.task_classes)
        self.assertIs(task_registry.get_task_class("ybt_arm_walk"), YBTArmWalkRobot)
        self.assertTrue(issubclass(YBTArmWalkRobot, YBTArmScriptedRobot))
        self.assertTrue(issubclass(YBTArmWalkRobot, YBTArmRobot))
        env_cfg, train_cfg = task_registry.get_cfgs("ybt_arm_walk")
        # the observation width is what makes a ybt_arm checkpoint loadable at all
        self.assertEqual((env_cfg.env.num_actions, env_cfg.env.num_observations), (18, 66))
        self.assertEqual(train_cfg.runner.experiment_name, "flat_ybt_arm_walk")
        self.assertFalse(train_cfg.runner.resume)

    def test_walking_commands_are_the_walking_task_s(self):
        # The whole point of the task: the commands are *not* pinned to zero.
        walking = YBTFlatArmCfg().commands
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"):
            self.assertEqual(getattr(self.cfg.commands.ranges, name),
                             getattr(walking.ranges, name), name)
        self.assertEqual(self.cfg.commands.ranges.lin_vel_x, [-0.8, 0.8])
        self.assertNotEqual(self.cfg.commands.ranges.lin_vel_x, [0.0, 0.0])
        self.assertTrue(self.cfg.commands.heading_command)
        self.assertFalse(self.cfg.commands.curriculum)

    def test_walking_rewards_and_push_are_inherited(self):
        # Redeclaring `rewards` wholesale is the easy way to lose these; assert the
        # inherited numbers rather than "some value".
        self.assertEqual(self.cfg.rewards.scales.feet_clearance, 0.3)
        self.assertTrue(self.cfg.domain_rand.push_robots)
        self.assertEqual(self.cfg.rewards.base_height_target,
                         YBTFlatArmCfg().rewards.base_height_target)
        self.assertEqual(self.cfg.env.episode_length_s,
                         YBTFlatArmCfg().env.episode_length_s)
        # and the standing task's overrides of the same two fields did not leak in
        self.assertEqual(YBTFlatArmStandCfg().rewards.scales.feet_clearance, 0.0)
        self.assertFalse(YBTFlatArmStandCfg().domain_rand.push_robots)

    def test_stand_only_fields_are_absent(self):
        self.assertFalse(hasattr(self.cfg, "stand"))
        self.assertTrue(hasattr(YBTFlatArmStandCfg(), "stand"))

    def test_arm_trajectory_is_not_on_the_frozen_task(self):
        # play.py's --arm_freq and check_ybt.py decide "is this a scripted-arm task?"
        # with hasattr(cfg, "arm_trajectory") / hasattr(env, "set_arm_scale").  Putting
        # the block on YBTFlatArmCfg would make the frozen-arm ybt_arm task look
        # scripted, and the flag would silently do nothing there.
        self.assertFalse(hasattr(YBTFlatArmCfg, "arm_trajectory"))
        self.assertTrue(hasattr(YBTFlatArmScriptedCfg, "arm_trajectory"))

    def test_scripted_config_is_shared_not_duplicated(self):
        walk, stand = self.cfg.arm_trajectory, YBTFlatArmStandCfg().arm_trajectory
        for name in ("operation_pose", "frequency", "curriculum_start", "curriculum_max",
                     "curriculum_step", "curriculum_interval_s",
                     "curriculum_min_episodes", "min_episode_fraction"):
            self.assertEqual(getattr(walk, name), getattr(stand, name), name)
        # the arm gains are the moving-arm gains, not the frozen-arm ones
        walking = YBTFlatArmCfg().control
        for name in ARM:
            self.assertAlmostEqual(self.cfg.control.stiffness[name],
                                   2.0 * walking.stiffness[name], places=6, msg=name)
            self.assertAlmostEqual(self.cfg.control.damping[name],
                                   0.25 * walking.damping[name], places=6, msg=name)
        # exactly the two gates differ, and in the safe direction for a walking gait
        self.assertGreater(walk.max_tilt_cost, stand.max_tilt_cost)
        self.assertLess(walk.min_tracking_reward, stand.min_tracking_reward)
        for name in ("min_episode_fraction", "curriculum_step"):
            self.assertEqual(getattr(walk, name), getattr(stand, name), name)

    def test_every_nonzero_reward_scale_has_a_function(self):
        for name, scale in all_scales(self.cfg).items():
            if name == "termination" or not isinstance(scale, (int, float)) or scale == 0:
                continue
            self.assertTrue(hasattr(YBTArmWalkRobot, "_reward_" + name), name)

    def test_curriculum_keys_are_live(self):
        # Both terms the ramp reads must survive the zero-scale pop; a zero here would
        # be a silently disabled curriculum.
        self.assertEqual(self.cfg.rewards.scales.tracking_lin_vel, 1.0)
        self.assertEqual(self.cfg.rewards.scales.orientation, -1.0)
        robot = make_robot(cfg=self.cfg, cls=YBTArmWalkRobot)
        robot.episode_sums = {key: torch.zeros(2)
                              for key in ("tracking_lin_vel", "orientation")}
        with patch.object(LeggedRobot, "_prepare_reward_function", lambda self: None):
            robot._prepare_reward_function()      # no raise
        for missing in ({"tracking_lin_vel"}, {"orientation"}, set()):
            robot.episode_sums = {key: torch.zeros(2) for key in missing}
            with patch.object(LeggedRobot, "_prepare_reward_function", lambda self: None):
                with self.assertRaisesRegex(ValueError, "non-zero scales"):
                    robot._prepare_reward_function()

    # --- the class hierarchy --------------------------------------------------
    def test_shared_machinery_is_not_duplicated(self):
        for name in SHARED_METHODS:
            self.assertIs(getattr(YBTArmWalkRobot, name),
                          getattr(YBTArmScriptedRobot, name), name)
            self.assertIs(getattr(YBTArmStandRobot, name),
                          getattr(YBTArmScriptedRobot, name), name)
        self.assertIs(YBTArmWalkRobot.CURRICULUM_KEYS,
                      YBTArmScriptedRobot.CURRICULUM_KEYS)
        # the classes own nothing of their own except the stand drift rule -- if a
        # method is ever copy-pasted into one of them, this fails
        own_walk = {k for k in YBTArmWalkRobot.__dict__ if not k.startswith("__")}
        own_stand = {k for k in YBTArmStandRobot.__dict__ if not k.startswith("__")}
        self.assertEqual(own_walk, set())
        self.assertEqual(own_stand, {"check_termination"})

    def test_termination_is_the_base_rule(self):
        # A walking robot is supposed to leave its origin, so the standing drift rule
        # must not apply.  This is arranged purely by the class hierarchy.
        self.assertIs(YBTArmWalkRobot.check_termination, LeggedRobot.check_termination)
        self.assertIsNot(YBTArmStandRobot.check_termination, LeggedRobot.check_termination)

        robot = make_robot(cfg=self.cfg, cls=YBTArmWalkRobot)
        robot.reset_buf = torch.zeros(robot.num_envs, dtype=torch.bool)
        robot.contact_forces = torch.zeros(robot.num_envs, 1, 3)
        robot.termination_contact_indices = torch.tensor([0])
        robot.rpy = torch.zeros(robot.num_envs, 3)
        robot.episode_length_buf = torch.zeros(robot.num_envs, dtype=torch.long)
        robot.base_pos = torch.zeros(robot.num_envs, 3)
        robot.env_origins = torch.zeros(robot.num_envs, 3)
        robot.base_pos[1, 0] = 2.0          # far from its slot, but level, no contact
        robot.check_termination()
        self.assertFalse(robot.reset_buf.any().item())

    # --- the scripted reference (spot checks; the shared class has the full set) ---
    def test_zero_amplitude_is_the_frozen_arm_task(self):
        robot = make_robot(cfg=self.cfg, cls=YBTArmWalkRobot)
        robot.arm_scale = 0.0
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot._refresh_arm_target()
        actions = torch.rand(robot.num_envs, robot.num_actions) * 2.0 - 1.0
        self.assertTrue(torch.equal(robot.arm_target,
                                    robot.arm_home.unsqueeze(0).repeat(robot.num_envs, 1)))
        self.assertTrue(torch.equal(robot._compute_torques(actions),
                                    LeggedRobot._compute_torques(robot, actions)))

    def test_set_arm_scale_repins_the_reference(self):
        robot = make_robot(cfg=self.cfg, cls=YBTArmWalkRobot)
        robot.arm_phase = torch.full((robot.num_envs,), math.pi)
        robot.set_arm_scale(self.cfg.arm_trajectory.curriculum_max)
        expected = arm_pose(self.cfg, self.cfg.arm_trajectory.operation_pose)
        self.assertTrue(torch.allclose(robot.arm_target,
                                       expected.unsqueeze(0).expand(robot.num_envs, len(ARM)),
                                       atol=1e-5))
        robot.set_arm_scale(0.0)
        self.assertTrue(torch.equal(robot.arm_target,
                                    robot.arm_home.unsqueeze(0).repeat(robot.num_envs, 1)))

    # --- the curriculum -------------------------------------------------------
    def prepare_curriculum(self, episodes=256, length=1000, track=0.9, tilt=0.0):
        robot = make_robot(cfg=self.cfg, cls=YBTArmWalkRobot)
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

    def test_curriculum_grows_on_a_healthy_walk(self):
        robot = self.prepare_curriculum()
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale, self.cfg.arm_trajectory.curriculum_step)
        self.assertEqual(robot._arm_stat["episodes"].item(), 0.0)

    def test_curriculum_does_not_outrun_its_interval(self):
        robot = self.prepare_curriculum()
        robot._update_arm_curriculum()
        robot._arm_stat["episodes"] += 256
        robot._arm_stat["lengths"] += 256 * 1000
        robot._arm_stat["tracking_lin_vel"] += 256 * 0.9
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale, self.cfg.arm_trajectory.curriculum_step)
        robot.common_step_counter += int(
            self.cfg.arm_trajectory.curriculum_interval_s / robot.dt)
        robot._update_arm_curriculum()
        self.assertAlmostEqual(robot.arm_scale,
                               2 * self.cfg.arm_trajectory.curriculum_step)

    def test_curriculum_holds_on_each_walking_gate(self):
        trajectory = self.cfg.arm_trajectory
        cases = {
            "episode ended early": {"length": 500.0},
            "stopped following the command":
                {"track": trajectory.min_tracking_reward - 0.05},
            # orientation counts down, so a more negative average means more tilt
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
        robot = self.prepare_curriculum()
        robot.arm_scale = trajectory.curriculum_max
        robot._update_arm_curriculum()
        self.assertEqual(robot.arm_scale, trajectory.curriculum_max)


if __name__ == "__main__":
    unittest.main()
