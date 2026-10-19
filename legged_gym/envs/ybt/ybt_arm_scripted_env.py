"""The scripted arm: its PD reference moves, the legs are the policy's business.

Shared by the two scripted-arm tasks -- `ybt_arm_stand` (legs hold the trunk still) and
`ybt_arm_walk` (legs track velocity commands) -- because everything here is about the
arm, not about what the legs are asked to do.  The policy still has all 18 outputs and
the arm's six are still ignored (`control.arm_action_scale` is 0.0); what changes is
that the arm's PD *reference* moves.

How the script reaches the arm
------------------------------
`LeggedRobot._compute_torques` computes

    torques = p * (action * action_scale + default_dof_pos - dof_pos) - d * dof_vel

and the arm's `action_scale` is zero, so its torque is a PD hold at `default_dof_pos`.
Moving what that hold is aiming at is the same as adding

    p * (reference - default_dof_pos)

to those six torques -- but only in exact arithmetic.  The base class clips at the effort
limit, and `p * (default_dof_pos - dof_pos)` is already past that limit whenever the arm
is more than a fraction of a radian off the folded pose, so `clip(base) + bias` is not
`clip(base + bias)`.  With the additive form an arm sitting a tenth of a radian below the
target it is heading for gets commanded to its *negative* limit and runs away to the joint
stop -- measured on the first version of this file: the elbow pinned at its limit with a
4.2 rad tracking error, arm torques at 100%, the trunk leaning 9 degrees.  So
`_compute_torques` below recomputes the six arm columns from the reference instead: the
same expression, evaluated before the clip.  The legs are untouched and still go through
the base class.

Why not just write the moving reference into `default_dof_pos`, which would be less
code?  Because `default_dof_pos` is what the observation block is measured against:

    obs_dof_pos = (dof_pos - default_dof_pos) * obs_scales.dof_pos

Moving it turns that block into the arm's *tracking error* -- a few hundredths of a
radian, the same size as the 0.01 observation noise -- so the policy would be blind to
the arm it is meant to be balancing against.  Left alone, the same block reads as the
arm's actual deviation from the folded pose (up to 2.1 rad at full amplitude) and the
policy can see the swing coming.  It also keeps "home" meaning the folded pose for
`_reset_dofs` and for anything else that asks.

The arm's *velocity* observations are useless either way (`obs_scales.dof_vel` is 0.05,
so 1-2 rad/s of arm motion lands under a noise scale of 1.5).  That is fine: the motion
is a known deterministic function of the phase, so seeing where the arm is tells the
policy where it is going.
"""
import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.ybt.ybt_arm_env import YBTArmRobot


class YBTArmScriptedRobot(YBTArmRobot):
    # reward terms the arm curriculum reads off finished episodes.  `orientation` is
    # -sum(gravity_xy^2), so its per-second average is the negative of the mean
    # squared tilt and the threshold in _update_arm_curriculum is compared with the
    # sign flipped.
    CURRICULUM_KEYS = ("tracking_lin_vel", "orientation")

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # episode_sums only holds the rewards that survived the zero-scale pop, and the
        # curriculum grows on the two below -- fail here rather than at the first reset
        missing = [key for key in self.CURRICULUM_KEYS if key not in self.episode_sums]
        if missing:
            raise ValueError(
                "the scripted-arm curriculum measures {}; give them non-zero scales in "
                "cfg.rewards.scales, or change CURRICULUM_KEYS".format(missing))

    def _init_buffers(self):
        if self.cfg.control.control_type != "P":
            raise NotImplementedError(
                "this task recomputes the arm's torque from the reference using the P law; "
                "cfg.control.control_type is {!r}".format(self.cfg.control.control_type))
        super()._init_buffers()
        trajectory = self.cfg.arm_trajectory
        self.arm_idx = self._arm_dof_indices()
        self.arm_home = self.default_dof_pos[0, self.arm_idx].clone()
        working = torch.tensor(
            [float(self._arm_angle(trajectory.operation_pose, self.dof_names[i]))
             for i in self.arm_idx],
            device=self.device)
        self.arm_swing = working - self.arm_home
        self.arm_target = self.arm_home.unsqueeze(0).repeat(self.num_envs, 1)
        # the arm's own slice of the PD, so the substep loop does not re-index
        self.arm_p_gains = self.p_gains[self.arm_idx].clone()
        self.arm_d_gains = self.d_gains[self.arm_idx].clone()
        self.arm_limits = self.torque_limits[self.arm_idx].clone()
        # One phase per env, so the 12288 robots sit spread through the working cycle
        # instead of moving in lockstep: a rollout only covers 24 * 20 ms = 0.48 s of a
        # 2 s cycle, and with a single shared phase every batch would see the same
        # quarter of it and the advantage estimate would have no phase diversity.
        # Never re-drawn at reset -- the phase is a running clock, not an episode state,
        # so an env that dies early does not replay the same opening every time.
        self.arm_phase = torch.rand(self.num_envs, device=self.device) * 2.0 * math.pi
        self.arm_scale = float(trajectory.curriculum_start)
        self._arm_reviewed = -10 ** 9
        self._reset_arm_stat()
        self._refresh_arm_target()

    @staticmethod
    def _arm_angle(pose, name):
        try:
            return pose[name]
        except KeyError:
            raise KeyError(
                "cfg.arm_trajectory.operation_pose has no entry for {}; it must name "
                "every arm joint".format(name)) from None

    def _reset_arm_stat(self):
        self._arm_stat = {key: torch.zeros((), device=self.device)
                          for key in ("episodes", "lengths") + self.CURRICULUM_KEYS}

    # --- the scripted reference ----------------------------------------------
    def _refresh_arm_target(self):
        """q = home + scale * (working - home) * (1 - cos(phase)) / 2."""
        ease = (1.0 - torch.cos(self.arm_phase)) * 0.5
        self.arm_target[:] = self.arm_home.unsqueeze(0) + (
            self.arm_scale * ease).unsqueeze(1) * self.arm_swing.unsqueeze(0)

    def set_arm_scale(self, scale):
        """Pin the difficulty instead of leaving it to the curriculum.

        `play.py` and `check_ybt.py` call this: `arm_scale` starts at
        `curriculum_start` (0.0, the folded pose) and only the training loop ever
        grows it, so without this a loaded policy is shown a motionless arm.
        """
        self.arm_scale = float(scale)
        self._refresh_arm_target()

    def step(self, actions):
        """Advance the working motion, then simulate.

        Once per policy step, not once per physics substep, and before the base class
        integrates anything, so the joints act on the reference from this instant
        rather than one step stale.  The 4 substeps of a step share one reference --
        a 20 ms staircase under a 2 s sine.
        """
        self.arm_phase = torch.remainder(
            self.arm_phase + 2.0 * math.pi * self.cfg.arm_trajectory.frequency * self.dt,
            2.0 * math.pi)
        self._refresh_arm_target()
        return super().step(actions)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        # Skipped at scale 0 so the frozen-arm phase is bit-identical to the walking
        # task, which is exactly the state a resumed frozen-arm checkpoint starts from.
        if self.arm_scale != 0.0:
            torques[:, self.arm_idx] = torch.clip(
                self.arm_p_gains * (self.arm_target - self.dof_pos[:, self.arm_idx])
                - self.arm_d_gains * self.dof_vel[:, self.arm_idx],
                -self.arm_limits, self.arm_limits)
        return torques

    # --- resets --------------------------------------------------------------
    def _reset_dofs(self, env_ids):
        """YBTRobot's reset, except the arm starts at its reference, not at home.

        At full amplitude the reference is up to 2.1 rad from the folded pose, and
        p_gains * that is ~800 N m against a 60 N m limit: spawning at home would slam
        every arm joint into its torque limit for the first couple of steps of every
        episode, which is an artefact of the reset rather than anything the task asks
        the legs to handle.
        """
        noise = self.cfg.init_state.joint_angle_noise
        positions = self.default_dof_pos + torch_rand_float(
            -noise, noise, (len(env_ids), self.num_dof), device=self.device)
        positions[:, self.arm_idx] = self.arm_target[env_ids]
        self.dof_pos[env_ids] = torch.maximum(
            torch.minimum(positions, self.dof_pos_limits[:, 1]),
            self.dof_pos_limits[:, 0])
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._collect_arm_stat(env_ids)
        super().reset_idx(env_ids)
        self._update_arm_curriculum()
        # after super(), which builds extras["episode"] from scratch
        self.extras["episode"]["arm_scale"] = float(self.arm_scale)

    def _collect_arm_stat(self, env_ids):
        """Accumulate finished-episode statistics, before the base class wipes them.

        reset_idx zeroes `episode_length_buf` and `episode_sums` for these envs, so
        this has to read them first.  Resets are staggered -- a call carries ~12 envs
        -- so the numbers accumulate across calls and are only judged once every
        `curriculum_interval_s`.

        The ramp therefore never starts on the first review.  `learn()` is called with
        `init_at_random_ep_len`, so the episodes that finish in the first seconds are
        whatever was left of a randomly chopped episode: too short to pass the
        `min_episode_fraction` test, and rightly so, since they say nothing about the
        task.  Measured at 4096 envs, the amplitude sat at 0 for the first 36
        iterations (17 s of simulated time) and then grew at every review.
        """
        lengths = self.episode_length_buf[env_ids].to(torch.float32)
        # envs that have never been stepped (the initial reset of everything) carry no
        # information, and averaging their zero-length episodes in would veto the first
        # review for no reason
        keep = lengths > 0.0
        ids, lengths = env_ids[keep], lengths[keep]
        self._arm_stat["episodes"] += ids.numel()
        self._arm_stat["lengths"] += lengths.sum()
        for key in self.CURRICULUM_KEYS:
            # per second of the episode that actually happened, which is what the
            # training log prints for a full-length episode
            self._arm_stat[key] += (self.episode_sums[key][ids]
                                    / (lengths * self.dt)).sum()

    def _update_arm_curriculum(self):
        trajectory = self.cfg.arm_trajectory
        interval = max(1, int(trajectory.curriculum_interval_s / self.dt))
        if self.common_step_counter - self._arm_reviewed < interval:
            return
        self._arm_reviewed = self.common_step_counter
        episodes = int(self._arm_stat["episodes"].item())
        if episodes < trajectory.curriculum_min_episodes:
            return
        mean_length = (self._arm_stat["lengths"] / episodes).item()
        mean_track = (self._arm_stat["tracking_lin_vel"] / episodes).item()
        mean_tilt = (self._arm_stat["orientation"] / episodes).item()
        self._reset_arm_stat()
        # `orientation` counts down, so its average is the negative of the mean squared
        # tilt and "tilt cost below the limit" is a >= on the negated value
        if (mean_length >= trajectory.min_episode_fraction * self.max_episode_length
                and mean_track >= trajectory.min_tracking_reward
                and -mean_tilt <= trajectory.max_tilt_cost):
            self.arm_scale = min(trajectory.curriculum_max,
                                 self.arm_scale + trajectory.curriculum_step)
