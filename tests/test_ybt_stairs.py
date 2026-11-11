"""Staircase terrain/config/reward regression checks. Run in the unitree-rl environment.

These tests do not build a simulation: they exercise the analytic geometry, the
task config, and the reward terms that were rewritten for stepped ground. The
environment itself is checked by running the task -- see
``doc/ybt_stairs_training_zh.md``.
"""
import os
import unittest

import isaacgym  # must precede torch
import numpy as np
import torch
from isaacgym import terrain_utils

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg
from legged_gym.envs.ybt.ybt_env import validate_ybt_asset
from legged_gym.envs.ybt.ybt_stairs_config import (
    StairsTerrain, StairsTerrainCfg, TiledStairsTerrain, YBTStairsCfg,
    YBTStairsCfgPPO)
from legged_gym.envs.ybt.ybt_stairs_env import YBTStairsRobot


class StairsTerrainTests(unittest.TestCase):
    """The single-tile geometry everything else is built from."""

    def setUp(self):
        self.cfg = StairsTerrainCfg()
        self.terrain = StairsTerrain(self.cfg)

    def test_layout_matches_the_configured_dimensions(self):
        t = self.terrain
        self.assertAlmostEqual(t.x_first_riser, self.cfg.approach_depth)
        self.assertAlmostEqual(
            t.x_top, self.cfg.approach_depth
            + self.cfg.num_steps * self.cfg.step_depth)
        self.assertAlmostEqual(
            t.top_height, self.cfg.num_steps * self.cfg.step_height)
        # flat run-up, steps in between, flat platform on top
        self.assertEqual(t.level_at(0.0), 0.0)
        self.assertAlmostEqual(t.level_at(t.x_first_riser + 1e-9),
                               self.cfg.step_height)
        self.assertAlmostEqual(t.level_at(t.x_top - 1e-9), t.top_height)
        self.assertAlmostEqual(t.level_at(self.cfg.terrain_length), t.top_height)

    def test_height_field_encodes_the_profile_exactly(self):
        # reconstruct the ground profile from the array and compare it against
        # the analytic one: this is what catches a slipped riser loop
        hf = self.terrain.height_field()
        self.assertEqual(hf.dtype, np.int16)
        self.assertEqual(hf.shape, (self.terrain.rows, self.terrain.cols))
        step = self.cfg.horizontal_scale
        # sample inside the staircase strip; the ground either side of it is
        # deliberately flat at every x
        y_lo = int(round(self.terrain.y_lo / step))
        y_hi = int(round(self.terrain.y_hi / step)) - 1
        # and sample cell centres, not edges: a riser sits exactly on a cell
        # boundary, so probing there asks which side of a tie a float rounds to
        for i in range(self.terrain.rows):
            x = (i + 0.5) * step
            for y in (y_lo, (y_lo + y_hi) // 2, y_hi):
                self.assertAlmostEqual(hf[i, y] * self.cfg.vertical_scale,
                                       self.terrain.level_at(x), places=9,
                                       msg="cell {} (x = {:.3f})".format(i, x))

    def test_quantisation_is_exact(self):
        # the analytic reward geometry is only reproducible if every configured
        # length lands on a whole number of cells; snapping is fine, drift is not
        t = self.terrain
        self.assertAlmostEqual(t.step_depth, self.cfg.step_depth)
        self.assertAlmostEqual(t.step_height, self.cfg.step_height)
        self.assertAlmostEqual(t.x_first_riser, self.cfg.approach_depth)

    def test_flat_tile_is_allowed(self):
        # level 0 of the map is flat, so step_height == 0 has to be legal
        flat = StairsTerrain(StairsTerrainCfg.with_overrides(step_height=0.0))
        self.assertEqual(flat.top_height, 0.0)
        self.assertFalse(flat.height_field().any())

    def test_narrow_staircase_leaves_flat_ground_either_side(self):
        terrain = StairsTerrain(StairsTerrainCfg.with_overrides(
            terrain_width=4.0, stair_width=2.0))
        hf = terrain.height_field()
        self.assertFalse(hf[:, 0].any())
        self.assertFalse(hf[:, -1].any())
        self.assertEqual(hf[:, hf.shape[1] // 2].max(),
                         self.cfg.num_steps * terrain.riser_units)

    def test_invalid_parameters_are_rejected(self):
        bad = {
            "num_steps": 0,
            "step_height": -0.1,
            "step_depth": 0.0,
            "approach_depth": -1.0,
            "platform_depth": 0.0,
        }
        for key, value in bad.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    StairsTerrain(StairsTerrainCfg.with_overrides(**{key: value}))
        for key, value in (("step_height", 0.001),   # smaller than one cell
                           ("step_depth", 0.01),
                           ("stair_width", 99.0)):   # wider than the tile
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    StairsTerrain(StairsTerrainCfg.with_overrides(**{key: value}))
        with self.assertRaises(ValueError):          # does not fit in the tile
            StairsTerrain(StairsTerrainCfg.with_overrides(
                approach_depth=10.0, terrain_length=12.0))

    def test_with_overrides_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            StairsTerrainCfg.with_overrides(num_risers=9)


class TiledStairsTerrainTests(unittest.TestCase):
    """The map the trainer lays down."""

    def setUp(self):
        self.cfg = YBTStairsCfg.terrain()
        self.map = TiledStairsTerrain(self.cfg)

    def test_map_shape_and_lane_grid(self):
        hf = self.map.height_field()
        self.assertEqual(hf.shape, (self.map.tile_rows * self.cfg.num_levels,
                                    self.map.tile_cols * self.cfg.num_lanes))
        origins = self.map.origins()
        self.assertEqual(origins.shape,
                         (self.cfg.num_levels, self.cfg.num_lanes, 3))
        ys = origins[0, :, 1]
        self.assertTrue(np.allclose(np.diff(ys), self.cfg.lane_width))
        self.assertGreater(ys[0], 0.0)
        self.assertLess(ys[-1], self.cfg.num_lanes * self.cfg.lane_width)
        xs = origins[:, 0, 0]
        self.assertTrue(np.allclose(np.diff(xs), self.map.tile_length))

    def test_every_spawn_stands_on_flat_ground(self):
        hf = self.map.height_field()
        origins = self.map.origins()
        for level in range(self.cfg.num_levels):
            for lane in (0, self.cfg.num_lanes // 2, self.cfg.num_lanes - 1):
                x, y = origins[level, lane, :2]
                cell = hf[int(round(x / self.cfg.horizontal_scale)),
                          int(round(y / self.cfg.horizontal_scale))]
                self.assertEqual(cell, 0,
                                 "spawn of level {} lane {} is not on the run-up"
                                 .format(level, lane))

    def test_difficulty_spread_runs_from_flat_to_full_height(self):
        heights = self.map.level_step_heights()
        self.assertEqual(len(heights), self.cfg.num_levels)
        self.assertAlmostEqual(heights[0], 0.0)
        self.assertAlmostEqual(heights[-1], self.cfg.step_height)
        self.assertTrue(np.all(np.diff(heights) > 0))

    def test_tile_budget_covers_the_configured_environments(self):
        self.assertGreaterEqual(self.map.num_tiles, YBTStairsCfg.env.num_envs)

    def test_hardest_risers_are_vertical_faces(self):
        """The converter must snap 15 cm risers, not leave them as steep ramps.

        Its test is a strict ``>`` on raw height units, so a threshold that
        happens to equal the riser slope never fires: the old 1.5 gave exactly
        30.000000000000004 units for a 30-unit riser and every step stayed a
        56.3 deg ramp, steeper than the terrain's 45 deg friction angle.
        """
        cfg = self.cfg
        hf = self.map.tiles[-1].height_field()

        def steepest_face(slope_threshold):
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                hf, horizontal_scale=cfg.horizontal_scale,
                vertical_scale=cfg.vertical_scale,
                slope_threshold=slope_threshold)
            a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
            b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
            normal = np.cross(a, b)
            normal = normal[np.linalg.norm(normal, axis=1) > 0]  # skip degenerates
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            return float(np.degrees(np.arccos(np.abs(normal[:, 2]))).max())

        self.assertGreater(steepest_face(cfg.slope_threshold), 85.0)
        # and pin down the bug itself, so re-raising the threshold fails loudly
        self.assertLess(steepest_face(1.5), 60.0)

    def test_describe_is_text(self):
        self.assertIn("levels", self.map.describe())


class _Stub(object):
    """Enough of an environment for the plain-tensor reward terms.

    The rewritten rewards only read ``root_states``/``rigid_body_states``/
    ``contact_forces``, so they can be exercised without a simulation.
    """

    _local_ground_heights = YBTStairsRobot._local_ground_heights
    _local_x = YBTStairsRobot._local_x
    _reward_base_height = YBTStairsRobot._reward_base_height
    _reward_feet_clearance = YBTStairsRobot._reward_feet_clearance
    _reward_feet_air_time = YBTStairsRobot._reward_feet_air_time
    _reward_progress = YBTStairsRobot._reward_progress
    _reward_reach_top = YBTStairsRobot._reward_reach_top
    _reward_heading = YBTStairsRobot._reward_heading
    _reward_lateral_vel = YBTStairsRobot._reward_lateral_vel
    _reward_lane = YBTStairsRobot._reward_lane
    _reward_speed_limit = YBTStairsRobot._reward_speed_limit
    _past_top_of_stairs = YBTStairsRobot._past_top_of_stairs

    def __init__(self, step_height):
        self.cfg = YBTStairsCfg()
        self.stairs = TiledStairsTerrain(self.cfg.terrain)
        self.feet_indices = torch.arange(4)
        self.env_step_height = torch.tensor([step_height])
        # tile 0 of level 0, i.e. the map origin sits at the run-up spawn point
        self.env_origins = torch.tensor(
            [[self.stairs.spawn_x, self.stairs.spawn_y, 0.0]])
        # policy step, exactly as LeggedRobot computes it
        self.dt = self.cfg.sim.dt * self.cfg.control.decimation
        self.num_envs = 1
        # vx, vy, yaw rate, target heading
        self.commands = torch.tensor([[0.4, 0.0, 0.0, 0.0]])
        self.rpy = torch.zeros(1, 3)
        self.base_lin_vel = torch.zeros(1, 3)
        self.root_states = torch.zeros(1, 13)
        self.rigid_body_states = torch.zeros(1, 4, 13)
        self.contact_forces = torch.zeros(1, 4, 3)
        self.last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        self.feet_air_time = torch.zeros(1, 4)
        self.last_local_x = self._local_x()
        self.episode_length_buf = torch.tensor([2])
        self.reached_top = torch.zeros(1, dtype=torch.bool)

    def local_x_to_world(self, local_x):
        """World x of a point standing local_x into its tile, as the env defines it."""
        return float(self.env_origins[0, 0]) + local_x - self.stairs.spawn_x

    def place(self, local_x):
        """Stand the trunk and all four feet local_x into the tile.

        Returns the ground height there, read back through the same call the
        rewards make -- so a test that compares against this value is comparing
        against the reward's own number rather than a second, differently
        rounded derivation of it.
        """
        world_x = self.local_x_to_world(local_x)
        self.root_states[0, 0] = world_x
        self.rigid_body_states[:, :, 0] = world_x
        return float(self._local_ground_heights(self.root_states[:, 0:1]))


class StairsRewardTests(unittest.TestCase):
    """The reward terms that assumed z = 0 was the ground."""

    def test_local_ground_height_follows_the_staircase(self):
        # the reference profile comes from the level-0 tile, scaled by each
        # level's riser height
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        hardest = stairs.tiles[-1]
        self.assertAlmostEqual(hardest.step_height, stairs.cfg.step_height)
        for step_height in (0.0, 0.05, 0.15):
            stub = _Stub(step_height)
            # offset off the cell boundary so a riser never lands exactly on a
            # sample and the comparison cannot hinge on a float tie
            for local_x in np.arange(0.017, stairs.tile_length, 0.05):
                steps = round(hardest.level_at(local_x) / hardest.step_height)
                self.assertAlmostEqual(
                    stub.place(local_x), steps * step_height, places=6,
                    msg="local_x = {:.2f}".format(local_x))

    def test_base_height_is_measured_from_the_step_under_the_trunk(self):
        # a trunk 0.5 m above the step it stands on is a perfect score, on the
        # run-up, on a mid-stair step, and on the platform of a 15 cm staircase
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        for step_height in (0.0, 0.15):
            for local_x in np.arange(0.017, stairs.tile_length, 0.25):
                stub = _Stub(step_height)
                ground = stub.place(local_x)
                stub.root_states[0, 2] = (ground
                                          + stub.cfg.rewards.base_height_target)
                self.assertAlmostEqual(float(stub._reward_base_height()), 0.0,
                                       places=9,
                                       msg="local_x = {:.2f}".format(local_x))

    def test_absolute_base_height_would_have_punished_the_climb(self):
        # guard the reason for the override: standing on the platform, the
        # inherited version is off by the whole climb height
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        top = stairs.num_steps * stairs.cfg.step_height
        self.assertAlmostEqual(stairs.tiles[-1].top_height, top)
        self.assertGreater(top, 0.5)  # bigger than the 0.5 m target itself

    def test_foot_clearance_is_measured_from_the_floor_under_the_trunk(self):
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        stub = _Stub(0.15)
        target = stub.cfg.rewards.feet_clearance_target
        cap = stub.cfg.rewards.feet_clearance_cap
        # trunk standing on the third step: a resting foot scores nothing, and
        # the reference follows the raised floor rather than the world origin
        ground = stub.place(stairs.x_first_riser + 2.5 * stairs.step_depth)
        self.assertGreater(ground, 0.0)  # it really is on a raised step
        stub.rigid_body_states[:, :, 2] = ground
        self.assertLess(float(stub._reward_feet_clearance()), 1e-9)
        # 17 cm above that floor is the neutral point, 27 cm is full marks
        stub.rigid_body_states[:, :, 2] = ground + target
        self.assertLess(float(stub._reward_feet_clearance()), 1e-9)
        stub.rigid_body_states[:, :, 2] = ground + cap
        self.assertAlmostEqual(float(stub._reward_feet_clearance()), 4.0,
                               places=6)
        # a foot already over the next tread must not be asked for the riser
        # height on top: trunk on step n, foot at the cap above the trunk's
        # step, i.e. 12 cm above the next (15 cm higher) tread.  The old
        # foot-referenced version saw 12 cm < its 17 cm target and scored 0.
        stub.rigid_body_states[:, :, 0] = stub.local_x_to_world(
            stairs.x_first_riser + 3.2 * stairs.step_depth)
        stub.rigid_body_states[:, :, 2] = ground + cap
        self.assertAlmostEqual(float(stub._reward_feet_clearance()), 4.0,
                               places=6)

    def test_foot_clearance_penalises_lifting_far_above_the_cap(self):
        stub = _Stub(0.0)
        target = stub.cfg.rewards.feet_clearance_target
        cap = stub.cfg.rewards.feet_clearance_cap
        width = cap - target
        # one full band width above the cap: neutral, two widths: negative
        stub.rigid_body_states[:, :, 2] = cap + width
        self.assertAlmostEqual(float(stub._reward_feet_clearance()), 0.0,
                               places=6)
        stub.rigid_body_states[:, :, 2] = cap + 2.0 * width
        self.assertAlmostEqual(float(stub._reward_feet_clearance()), -4.0,
                               places=6)

    def test_progress_pays_for_net_displacement_only(self):
        stub = _Stub(0.0)
        speed = float(stub.commands[0, 0])
        start = 1.0
        stub.last_local_x = torch.tensor([start])
        # one commanded step forward: exactly the tracking score
        stub.place(start + speed * stub.dt)
        # 1e-5 tolerance: the stub stores x in float32, so a 2 mm move carries a
        # few 1e-8 relative to the metre-scale coordinate
        self.assertAlmostEqual(float(stub._reward_progress()), 1.0, places=4)
        # no motion at all: nothing paid, even though velocity rewards might not
        # notice a robot treading against a riser
        stub.place(start + speed * stub.dt)
        self.assertAlmostEqual(float(stub._reward_progress()), 0.0, places=6)
        # sliding backwards is bounded, not unbounded
        stub.place(start - 10.0 * speed * stub.dt)
        self.assertAlmostEqual(float(stub._reward_progress()), -1.0, places=6)
        # and so is overspeeding: charging double would pay for a sprint, which
        # is exactly what the fine-tuning runs learned to do
        stub.last_local_x = torch.tensor([start])
        stub.place(start + 2.0 * speed * stub.dt)
        self.assertAlmostEqual(float(stub._reward_progress()), 1.0, places=4)

    def test_progress_ignores_the_reset_teleport(self):
        stub = _Stub(0.0)
        stub.episode_length_buf = torch.tensor([1])  # first step after a reset
        stub.place(3.0)
        self.assertEqual(float(stub._reward_progress()), 0.0)
        # but the reference still moved with the robot, so the next step is sane
        stub.episode_length_buf = torch.tensor([2])
        stub.place(3.0)
        self.assertEqual(float(stub._reward_progress()), 0.0)

    def test_heading_penalty_sees_a_static_yaw_offset(self):
        # tracking_ang_vel only sees the yaw *rate*; this term prices the offset
        stub = _Stub(0.0)
        stub.rpy[0, 2] = np.deg2rad(20.0)
        self.assertAlmostEqual(float(stub._reward_heading()),
                               np.deg2rad(20.0) ** 2, places=6)
        stub.rpy[0, 2] = 0.0
        self.assertEqual(float(stub._reward_heading()), 0.0)
        # the error wraps: 350 deg is 10 deg to the left, not 350 deg right
        stub.rpy[0, 2] = np.deg2rad(350.0)
        self.assertAlmostEqual(float(stub._reward_heading()),
                               np.deg2rad(10.0) ** 2, places=6)

    def test_lateral_velocity_is_the_crab_component(self):
        stub = _Stub(0.0)
        stub.base_lin_vel[0, 1] = -0.6
        self.assertAlmostEqual(float(stub._reward_lateral_vel()), 0.6, places=6)

    def test_lane_penalty_ignores_small_scatter_but_not_drift(self):
        stub = _Stub(0.0)
        centre = float(stub.env_origins[0, 1])
        stub.root_states[0, 1] = centre + 0.08   # inside the 0.10 m dead band
        self.assertEqual(float(stub._reward_lane()), 0.0)
        stub.root_states[0, 1] = centre + 0.60
        self.assertAlmostEqual(float(stub._reward_lane()), 0.25, places=6)

    def test_speed_limit_only_charges_for_overspeeding(self):
        stub = _Stub(0.0)
        cmd = float(stub.commands[0, 0])          # 0.4 m/s
        margin = stub.cfg.rewards.speed_margin
        # slower than commanded: this term is silent (tracking handles it)
        stub.base_lin_vel[0, 0] = cmd - 0.3
        self.assertEqual(float(stub._reward_speed_limit()), 0.0)
        # ordinary overshoot inside the margin: still free
        stub.base_lin_vel[0, 0] = cmd + margin
        self.assertAlmostEqual(float(stub._reward_speed_limit()), 0.0, places=6)
        # the sprint measured on 2026-10-19: 2.5 m/s against a 0.4 command
        stub.base_lin_vel[0, 0] = 2.5
        excess = 2.5 - cmd - margin
        self.assertAlmostEqual(float(stub._reward_speed_limit()),
                               excess ** 2, places=6)

    def test_reach_top_is_a_one_shot_bonus(self):
        stub = _Stub(0.15)
        self.assertEqual(float(stub._reward_reach_top()), 0.0)
        stub.reached_top[:] = True
        # `_prepare_reward_function` multiplies the scale by dt; dividing here
        # makes the configured 5.0 the real bonus
        self.assertAlmostEqual(float(stub._reward_reach_top()) * stub.dt, 1.0,
                               places=9)

    def test_feet_air_time_rebases_on_a_stair_length_stride(self):
        stub = _Stub(0.15)
        stub.feet_air_time[:] = 0.3       # a step period on 0.30 m treads
        stub.contact_forces[:, 0, 2] = 10.0  # FR just landed
        reward = float(stub._reward_feet_air_time())
        threshold = stub.cfg.rewards.feet_air_time_positive_threshold
        self.assertAlmostEqual(reward, 0.3 + stub.dt - threshold, places=6)
        self.assertGreater(reward, 0.0)   # the inherited 0.5 s would be negative


class _TerminationStub(YBTStairsRobot):
    """No simulation: just the state ``LeggedRobot.check_termination`` reads."""

    def __init__(self, past_top):
        self.cfg = YBTStairsCfg()
        self.contact_forces = torch.zeros(len(past_top), 1, 3)
        self.termination_contact_indices = torch.tensor([0])
        self.rpy = torch.zeros(len(past_top), 3)
        self.episode_length_buf = torch.zeros(len(past_top), dtype=torch.long)
        self.max_episode_length = 100
        self.reset_buf = torch.zeros(len(past_top), dtype=torch.bool)
        self.time_out_buf = torch.zeros(len(past_top), dtype=torch.bool)
        self._past = torch.tensor(past_top, dtype=torch.bool)

    def _past_top_of_stairs(self):
        return self._past


class StairsTerminationTests(unittest.TestCase):
    """Success is a truncation, and the base terminations still work."""

    def test_reaching_the_top_is_a_timeout_not_a_fall(self):
        stub = _TerminationStub([False, True])
        stub.check_termination()
        self.assertFalse(bool(stub.reset_buf[0]))
        self.assertTrue(bool(stub.reset_buf[1]))
        self.assertTrue(bool(stub.time_out_buf[1]))
        # PPO bootstraps on time-outs; a success must not read as a death
        self.assertTrue(bool(stub.reached_top[1]))
        self.assertFalse(bool(stub.reached_top[0]))

    def test_the_usual_fall_terminations_still_fire(self):
        stub = _TerminationStub([False, False])
        stub.rpy[0, 0] = 1.0  # |roll| > 0.8
        stub.check_termination()
        self.assertTrue(bool(stub.reset_buf[0]))
        self.assertFalse(bool(stub.time_out_buf[0]))
        self.assertFalse(bool(stub.reached_top[0]))


class StairsEnvContractTests(unittest.TestCase):
    """What the registry, the PPO runner and the flat task rely on."""

    def test_task_is_registered(self):
        self.assertIn("ybt_stairs", task_registry.task_classes)
        self.assertIs(task_registry.get_task_class("ybt_stairs"), YBTStairsRobot)
        _, ppo_cfg = task_registry.get_cfgs("ybt_stairs")
        self.assertIsInstance(ppo_cfg.runner, YBTStairsCfgPPO.runner)

    def test_flat_task_still_refuses_anything_but_a_plane(self):
        cfg = YBTFlatCfg()
        self.assertEqual(cfg.terrain.mesh_type, "plane")
        self.assertEqual(cfg.env.num_envs, 12288)
        # YBTFlatCfg.terrain is a class, so a plain attribute write would mutate
        # it for every later test in this process; subclass it instead
        cfg.terrain = type("terrain", (YBTFlatCfg.terrain,),
                           {"mesh_type": "trimesh"})
        with self.assertRaises(ValueError):
            validate_ybt_asset(cfg)
        self.assertEqual(YBTFlatCfg.terrain.mesh_type, "plane")
        # the staircase task passes the same validation with the flag off
        validate_ybt_asset(YBTStairsCfg(), flat_ground_only=False)

    def test_every_non_zero_reward_scale_names_a_real_function(self):
        # non-zero scales are resolved as getattr(env, "_reward_" + name), so a
        # name with no function is an AttributeError at construction time -- the
        # base cfg ships exactly such a name (`feet_stumble`) at zero scale
        scales = YBTStairsCfg().rewards.scales
        for name, scale in vars(scales).items():
            if name.startswith("_") or not isinstance(scale, (int, float)):
                continue
            if scale == 0:
                continue
            with self.subTest(scale=name):
                self.assertTrue(
                    hasattr(YBTStairsRobot, "_reward_" + name),
                    "reward scale '{}' has no _reward_{} implementation".format(
                        name, name))

    def test_stairs_logs_do_not_collide_with_the_flat_run(self):
        self.assertEqual(YBTStairsCfgPPO.runner.experiment_name, "stairs_ybt")
        self.assertNotEqual(YBTStairsCfgPPO.runner.experiment_name, "flat_ybt")
        self.assertFalse(YBTStairsCfgPPO.runner.resume)
        self.assertNotEqual(
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs",
                         YBTStairsCfgPPO.runner.experiment_name),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "flat_ybt"))

    def test_stairs_asks_for_more_clearance_than_flat(self):
        self.assertGreater(YBTStairsCfg().rewards.feet_clearance_target,
                           YBTFlatCfg().rewards.feet_clearance_target)

    def test_tile_count_bounds_the_environment_count(self):
        cfg = YBTStairsCfg()
        self.assertLessEqual(cfg.env.num_envs,
                             cfg.terrain.num_levels * cfg.terrain.num_lanes)

    def test_termination_flags_a_robot_past_the_last_riser(self):
        stub = _Stub(0.15)
        stairs = stub.stairs
        stub.root_states[0, 0] = stub.local_x_to_world(stairs.x_top - 0.1)
        self.assertFalse(bool(stub._past_top_of_stairs()))
        stub.root_states[0, 0] = stub.local_x_to_world(stairs.x_top + 0.1)
        self.assertTrue(bool(stub._past_top_of_stairs()))

    def test_stairs_robot_extends_the_flat_one(self):
        # the staircase task reuses the validated URDF, safe reset and rigid-body
        # bookkeeping of the flat task rather than reimplementing them
        from legged_gym.envs.ybt.ybt_env import YBTRobot
        self.assertTrue(issubclass(YBTStairsRobot, YBTRobot))
        self.assertTrue(issubclass(YBTStairsRobot, LeggedRobot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
