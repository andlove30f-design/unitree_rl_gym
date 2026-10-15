"""Armed-YBT regression checks: dimensions, the folded arm pose, and the freeze.

Run in the unitree-rl environment:

    python -m unittest discover -s tests -p "test_ybt*.py" -v
"""
import copy
import struct
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import isaacgym  # must precede torch
import numpy as np
import torch

try:  # `unittest discover -s tests` puts tests/ on the path, pytest may not
    from test_ybt import default_link_transforms
except ImportError:
    from tests.test_ybt import default_link_transforms

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfg, YBTFlatArmCfgPPO
from legged_gym.envs.ybt.ybt_arm_env import YBTArmRobot
from legged_gym.envs.ybt.ybt_env import YBTRobot, validate_ybt_asset

ARM_JOINTS = ("arm_joint_1", "arm_joint_2", "arm_joint_3",
              "arm_joint_4", "arm_joint_5", "arm_joint_6")


def arm_indices(names):
    return [i for i, name in enumerate(names) if name.startswith("arm_joint")]


def binary_stl_bounds(path):
    """(min, max) of the vertices of a binary STL, as numpy arrays."""
    data = path.read_bytes()
    facets = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + facets * 50:
        raise ValueError("{} is not a binary STL".format(path))
    vertices = np.empty((facets * 3, 3), dtype=np.float32)
    for facet in range(facets):
        offset = 84 + facet * 50 + 12
        for corner in range(3):
            vertices[facet * 3 + corner] = struct.unpack_from("<3f", data, offset + corner * 12)
    return vertices.min(axis=0), vertices.max(axis=0)


def collision_bounds(link, urdf_dir):
    """Bounding box of a link's collision geometry, in the link frame."""
    lows, highs = [], []
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        mesh = geometry.find("mesh")
        if mesh is not None:
            low, high = binary_stl_bounds(urdf_dir / mesh.attrib["filename"])
        elif geometry.find("box") is not None:
            half = np.fromstring(geometry.find("box").attrib["size"], sep=" ") / 2
            low, high = -half, half
        elif geometry.find("cylinder") is not None:
            radius = float(geometry.find("cylinder").attrib["radius"])
            length = float(geometry.find("cylinder").attrib["length"]) / 2
            low, high = np.array([-radius, -radius, -length]), np.array([radius, radius, length])
        else:
            radius = float(geometry.find("sphere").attrib["radius"])
            low, high = np.full(3, -radius), np.full(3, radius)
        # every collision origin in this asset is identity, but do not assume it
        origin = collision.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        lows.append(low + xyz)
        highs.append(high + xyz)
    return np.min(lows, axis=0), np.max(highs, axis=0)


class YBTArmTests(unittest.TestCase):
    def setUp(self):
        self.cfg = YBTFlatArmCfg()
        self.path = validate_ybt_asset(self.cfg)
        self.root = ET.parse(str(self.path)).getroot()

    def test_registration(self):
        self.assertIn("ybt_arm", task_registry.task_classes)
        self.assertIs(task_registry.get_task_class("ybt_arm"), YBTArmRobot)
        self.assertTrue(issubclass(YBTArmRobot, YBTRobot))
        self.assertTrue(issubclass(YBTRobot, LeggedRobot))
        self.assertEqual(task_registry.get_cfgs("ybt_arm")[1].runner.experiment_name,
                         "flat_ybt_arm")

    def test_training_defaults(self):
        ppo = YBTFlatArmCfgPPO()
        self.assertEqual(self.cfg.env.num_envs, 12288)
        self.assertEqual((self.cfg.env.num_actions, self.cfg.env.num_observations), (18, 66))
        self.assertEqual(self.cfg.sim.dt * self.cfg.control.decimation, 0.02)
        self.assertEqual(ppo.runner.experiment_name, "flat_ybt_arm")
        # the walking recipe is inherited, not retyped
        self.assertEqual(ppo.runner.num_steps_per_env, 24)
        self.assertEqual(ppo.runner.max_iterations, 1500)
        self.assertEqual(self.cfg.rewards.scales.feet_clearance, 0.3)
        self.assertEqual(self.cfg.rewards.scales.tracking_lin_vel, 1.0)
        self.assertEqual(self.cfg.rewards.base_height_target, 0.50)
        self.assertEqual(self.cfg.init_state.pos[2], 0.55)

    def test_only_the_arm_differs_from_the_armless_task(self):
        armless = YBTFlatCfg()
        for name, angle in armless.init_state.default_joint_angles.items():
            self.assertEqual(self.cfg.init_state.default_joint_angles[name], angle, name)
        self.assertEqual(set(self.cfg.init_state.default_joint_angles),
                         set(armless.init_state.default_joint_angles) | set(ARM_JOINTS))
        for key in ("hip_joint", "thigh_joint", "calf_joint"):
            self.assertEqual(self.cfg.control.stiffness[key], armless.control.stiffness[key])
            self.assertEqual(self.cfg.control.damping[key], armless.control.damping[key])
        self.assertEqual(self.cfg.control.action_scale, armless.control.action_scale)
        self.assertEqual(self.cfg.commands.ranges.lin_vel_x, armless.commands.ranges.lin_vel_x)
        self.assertEqual(copy.copy(self.cfg.asset.foot_name), armless.asset.foot_name)
        self.assertEqual(self.cfg.asset.terminate_after_contacts_on, ["trunk"])

    def test_arm_gains_cover_every_arm_joint(self):
        # `_init_buffers` matches gains by substring and a joint with no match is
        # left limp, which for a frozen arm means it just falls over
        for name in ARM_JOINTS:
            gains = [key for key in self.cfg.control.stiffness if key in name]
            self.assertEqual(gains, [name], name)
            self.assertEqual([key for key in self.cfg.control.damping if key in name],
                             [name], name)
            self.assertGreater(self.cfg.control.stiffness[name], 0.0)

    def test_arm_gains_scale_with_joint_inertia(self):
        # One scalar gain for all six does not work: k=150 chatters at the wrist
        # (mass-matrix diagonal ~0.002 kg m^2) and is limp at the shoulder (~0.13).
        # The gains must shrink by orders of magnitude going out along the arm.
        gains = [self.cfg.control.stiffness[name] for name in ARM_JOINTS]
        self.assertEqual(gains, sorted(gains, reverse=True))  # base -> wrist, softer
        self.assertGreater(gains[0], 50 * gains[5])
        # Both halves of the PD have a hard upper bound at a 200 Hz physics step, and
        # the wrists sit close to both of them. See ybt_arm_config.control.
        inertias = [0.15666, 0.12696, 0.08822, 0.00390, 0.00189, 0.0]
        armature = self.cfg.control.arm_armature
        for name, gain, inertia in zip(ARM_JOINTS, gains, inertias):
            motor = inertia + armature
            # stiffness: the position loop must not alias, w = sqrt(k / M), w dt < 2
            omega = (gain / motor) ** 0.5
            self.assertLess(omega * self.cfg.sim.dt, 0.5, name)
            # damping: v *= 1 - d dt / M, so d dt < 2 M or the damping term itself
            # overshoots. Raising d is not a safe way to quiet a ringing joint.
            self.assertLess(self.cfg.control.damping[name] * self.cfg.sim.dt, 2 * motor, name)

    def test_folded_arm_clears_the_trunk(self):
        # The held pose is the default pose, so the fold has to be clear of the
        # trunk's collision box or the arm would be jammed into the body.
        transforms = default_link_transforms(self.root, self.cfg.init_state.default_joint_angles)
        trunk = self.root.find("link[@name='trunk']")
        _, trunk_high = collision_bounds(trunk, self.path.parent)
        self.assertAlmostEqual(trunk_high[2], 0.09, places=6)  # 0.18 m box centred on the trunk

        lowest = None
        for name in ("arm_base_link",) + tuple(n for n in transforms if n.startswith("arm_link")):
            low, high = collision_bounds(self.root.find("link[@name='{}']".format(name)),
                                         self.path.parent)
            rotation = transforms[name][:3, :3]
            offset = transforms[name][:3, 3]
            corners = np.array([[x, y, z] for x in (low[0], high[0])
                                for y in (low[1], high[1]) for z in (low[2], high[2])])
            world = corners @ rotation.T + offset
            self.assertGreater(world[:, 2].min(), trunk_high[2], name)
            if name == "arm_base_link":
                continue  # bolted onto the deck, so it sits on it by design
            if lowest is None or world[:, 2].min() < lowest[0]:
                lowest = (world[:, 2].min(), name)
        # everything above the base plate floats clear of the deck instead of
        # resting on or through it
        self.assertGreater(lowest[0], trunk_high[2] + 0.03)
        self.assertGreater(lowest[0], 0.1012)  # the deck's own top face

    def test_arm_pose_is_folded_and_inside_the_limits(self):
        joints = {joint.attrib["name"]: joint for joint in self.root.findall("joint")}
        folded = 0
        for name in ARM_JOINTS:
            angle = self.cfg.init_state.default_joint_angles[name]
            limit = joints[name].find("limit")
            self.assertLess(float(limit.attrib["lower"]), angle)
            self.assertLess(angle, float(limit.attrib["upper"]))
            # stay clear of the limits so reset noise cannot push past them
            self.assertLess(abs(angle), max(abs(float(limit.attrib["lower"])),
                                            abs(float(limit.attrib["upper"]))) - 0.1)
            if angle != 0.0:
                folded += 1
        self.assertEqual(folded, 3)  # arm_joint_2/3/5, the rest centred

    def test_stand_has_four_feet_above_ground(self):
        transforms = default_link_transforms(self.root, self.cfg.init_state.default_joint_angles)
        self.assertEqual(len(transforms), len(self.root.findall("link")))
        for leg in ("FR", "FL", "RR", "RL"):
            foot = self.root.find("link[@name='{}_foot']".format(leg))
            radius = float(foot.find("collision/geometry/sphere").attrib["radius"])
            height = self.cfg.init_state.pos[2] + transforms[leg + "_foot"][2, 3]
            self.assertGreater(height - radius, 0.0)
            self.assertLess(height - radius, 0.10)
        # the folded arm is nowhere near the ground, so the spawn height and the
        # lowest collision point stay the armless robot's
        for name, transform in transforms.items():
            if name.startswith("arm"):
                self.assertGreater(self.cfg.init_state.pos[2] + transform[2, 3], 0.3)

    def test_mass_and_feet(self):
        mass = sum(float(link.find("inertial/mass").attrib["value"])
                   for link in self.root.findall("link"))
        self.assertAlmostEqual(mass, 68.842175, places=5)
        names = {link.attrib["name"] for link in self.root.findall("link")}
        feet = {name for name in names if self.cfg.asset.foot_name in name}
        self.assertEqual(feet, {"FR_foot", "FL_foot", "RR_foot", "RL_foot"})
        self.assertFalse(any(name.startswith("arm") for name in feet))

    def test_penalised_contacts_include_the_whole_arm(self):
        # "arm" has to match arm_base_link too, not just arm_link_*
        names = {link.attrib["name"] for link in self.root.findall("link")}
        matched = {name for name in names
                   for pattern in self.cfg.asset.penalize_contacts_on if pattern in name}
        self.assertEqual({name for name in matched if name.startswith("arm")},
                         {"arm_base_link"} | {n for n in names if n.startswith("arm_link")})
        self.assertEqual({name for name in matched if not name.startswith("arm")},
                         {n for n in names if "thigh" in n or "calf" in n})

    def test_rejects_observation_mismatch(self):
        bad = copy.deepcopy(self.cfg)
        bad.env.num_observations = 65
        with self.assertRaisesRegex(ValueError, "66 observations"):
            validate_ybt_asset(bad)

    def test_rejects_missing_arm_joints(self):
        bad = copy.deepcopy(self.cfg)
        bad.init_state.default_joint_angles = {
            k: v for k, v in self.cfg.init_state.default_joint_angles.items()
            if k != "arm_joint_5"}
        with self.assertRaisesRegex(ValueError, "17 movable joints"):
            validate_ybt_asset(bad)


class YBTArmFreezeTests(unittest.TestCase):
    """The arm must ignore the policy, hold its folded pose, and cost no torque."""

    def setUp(self):
        self.cfg = YBTFlatArmCfg()

    def bare_robot(self, names):
        """YBTArmRobot with the base _init_buffers stubbed out (no PhysX here)."""
        robot = object.__new__(YBTArmRobot)
        robot.cfg = self.cfg
        robot.device = "cpu"
        robot.num_dof = len(names)
        robot.dof_names = list(names)

        def stub(robot):
            # mirrors LeggedRobot._init_buffers' seeding of the per-joint scale
            robot.action_scale = torch.full(
                (robot.num_dof,), float(robot.cfg.control.action_scale))

        with patch.object(YBTRobot, "_init_buffers", stub):
            robot._init_buffers()
        return robot

    def test_arm_joints_are_frozen_legs_are_not(self):
        names = list(self.cfg.init_state.default_joint_angles)
        # deliberately not the URDF order: the override must go by name
        names = names[::-1]
        robot = self.bare_robot(names)
        scale = robot.action_scale.tolist()
        for index, name in enumerate(names):
            expected = self.cfg.control.arm_action_scale if name.startswith("arm_joint") else 0.25
            self.assertEqual(scale[index], expected, name)
        self.assertEqual(self.cfg.control.arm_action_scale, 0.0)

    def test_rejects_an_asset_without_six_arm_joints(self):
        names = [n for n in self.cfg.init_state.default_joint_angles
                 if not n.startswith("arm_joint")] + ["arm_joint_1"]
        with self.assertRaisesRegex(ValueError, "expects 6 arm joints"):
            self.bare_robot(names)

    def test_only_the_arm_dofs_get_rotor_inertia(self):
        # arm_joint_6's mass matrix diagonal is 0, which puts no bound at all on a
        # position gain, so the arm DOFs carry rotor inertia. The legs must not: the
        # calf diagonal is 0.00216, so armature there would soften the stance by ~28%
        # versus the armless task, which is the one thing this task must not do.
        names = list(self.cfg.init_state.default_joint_angles)[::-1]
        robot = self.bare_robot(names)
        props = {"lower": np.full(len(names), -2.0), "upper": np.full(len(names), 2.0),
                 "velocity": np.full(len(names), 30.0), "effort": np.full(len(names), 60.0),
                 "armature": np.zeros(len(names))}
        robot._process_dof_props(props, 0)
        self.assertGreater(self.cfg.control.arm_armature, 0.0)
        for index, name in enumerate(names):
            expected = self.cfg.control.arm_armature if name.startswith("arm_joint") else 0.0
            self.assertEqual(props["armature"][index], expected, name)

    def test_frozen_torque_ignores_the_action(self):
        names = list(self.cfg.init_state.default_joint_angles)
        robot = self.bare_robot(names)
        robot.num_envs = 1
        robot.torques = torch.zeros(1, robot.num_dof)
        robot.p_gains = torch.zeros(robot.num_dof)
        robot.d_gains = torch.zeros(robot.num_dof)
        for index, name in enumerate(names):
            for key in self.cfg.control.stiffness:
                if key in name:
                    robot.p_gains[index] = self.cfg.control.stiffness[key]
                    robot.d_gains[index] = self.cfg.control.damping[key]
        robot.default_dof_pos = torch.tensor(
            [[self.cfg.init_state.default_joint_angles[n] for n in names]])
        robot.dof_pos = robot.default_dof_pos.clone()
        robot.dof_vel = torch.zeros_like(robot.dof_pos)
        robot.torque_limits = torch.full_like(robot.dof_pos, 60.0)

        arm = arm_indices(names)
        legs = [i for i in range(len(names)) if i not in arm]
        zeros = torch.zeros(1, robot.num_dof)
        ones = torch.ones(1, robot.num_dof)
        torque_zeros = robot._compute_torques(zeros)
        torque_ones = robot._compute_torques(ones)

        # a frozen joint is a PD hold at the default angle: the action cannot move it
        self.assertEqual(torque_zeros[0, arm].abs().max().item(), 0.0)
        self.assertEqual(torque_ones[0, arm].abs().max().item(), 0.0)
        # and the legs still respond to the action
        self.assertTrue(torch.all(torque_zeros[0, legs] == 0.0))
        self.assertTrue(torch.all(torque_ones[0, legs] != 0.0))
        self.assertTrue(torch.allclose(torque_ones[0, legs],
                                       robot.p_gains[legs] * 0.25))

    def test_frozen_arm_holds_its_gravity_load_with_small_sag(self):
        # Folded, the arm puts 7.13 N.m on arm_joint_2: 3.80 kg of links at and below
        # that joint's child, over a 0.19 m lever (resources/robots/ybt_arm/tools/
        # folded_pose.py). A frozen joint can only answer that with position error, so
        # the shoulder gain has to be high enough to keep the sag small -- and low
        # enough to stay stable at 200 Hz, which is what the sweep found.
        names = list(self.cfg.init_state.default_joint_angles)
        shoulder = names.index("arm_joint_2")
        load = 7.13  # N.m, from the folded-pose forward kinematics
        stiffness = self.cfg.control.stiffness["arm_joint_2"]
        sag = load / stiffness
        self.assertLess(sag, 0.03)  # < 1.1 degrees of droop

        robot = self.bare_robot(names)
        robot.num_envs = 1
        robot.default_dof_pos = torch.tensor(
            [[self.cfg.init_state.default_joint_angles[n] for n in names]])
        robot.p_gains = torch.zeros(robot.num_dof)
        robot.d_gains = torch.zeros(robot.num_dof)
        for name in ARM_JOINTS:  # the legs are irrelevant here
            robot.p_gains[names.index(name)] = self.cfg.control.stiffness[name]
            robot.d_gains[names.index(name)] = self.cfg.control.damping[name]
        robot.torque_limits = torch.full((1, robot.num_dof), 1e3)
        robot.dof_vel = torch.zeros(1, robot.num_dof)
        robot.dof_pos = robot.default_dof_pos.clone()
        robot.dof_pos[0, shoulder] -= sag
        torque = robot._compute_torques(torch.zeros(1, robot.num_dof))
        self.assertAlmostEqual(torque[0, shoulder].item(), load, places=1)


if __name__ == "__main__":
    unittest.main()
