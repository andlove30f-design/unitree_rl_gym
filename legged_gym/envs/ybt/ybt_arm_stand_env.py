"""Standing balance: the arm follows a scripted motion, the legs hold the trunk.

The scripted-arm machinery lives in ybt_arm_scripted_env.py; this file adds only the
thing that makes the task "standing": a drift termination.
"""
import torch

from legged_gym.envs.ybt.ybt_arm_scripted_env import YBTArmScriptedRobot


class YBTArmStandRobot(YBTArmScriptedRobot):
    def check_termination(self):
        super().check_termination()
        # Standing includes staying put.  A slow unpunished drift is worth 4% of
        # tracking_lin_vel, so over 20 s the trunk can travel metres -- and env_spacing
        # is 3 m, so two robots would end up sharing a volume.  A failure, not a
        # timeout, so it is not counted as reaching the end of the episode.
        drift = torch.norm(self.base_pos[:, :2] - self.env_origins[:, :2], dim=1)
        self.reset_buf |= drift > self.cfg.stand.max_drift
