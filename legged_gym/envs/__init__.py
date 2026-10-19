from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.ybt.ybt_config import YBTFlatCfg, YBTFlatCfgPPO
from legged_gym.envs.ybt.ybt_env import YBTRobot
from legged_gym.envs.ybt.ybt_arm_config import YBTFlatArmCfg, YBTFlatArmCfgPPO
from legged_gym.envs.ybt.ybt_arm_env import YBTArmRobot
from legged_gym.envs.ybt.ybt_arm_stand_config import (YBTFlatArmStandCfg,
                                                      YBTFlatArmStandCfgPPO)
from legged_gym.envs.ybt.ybt_arm_stand_env import YBTArmStandRobot
from legged_gym.envs.ybt.ybt_arm_walk_config import (YBTFlatArmWalkCfg,
                                                     YBTFlatArmWalkCfgPPO)
from legged_gym.envs.ybt.ybt_arm_walk_env import YBTArmWalkRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "ybt", YBTRobot, YBTFlatCfg(), YBTFlatCfgPPO())
task_registry.register( "ybt_arm", YBTArmRobot, YBTFlatArmCfg(), YBTFlatArmCfgPPO())
task_registry.register( "ybt_arm_stand", YBTArmStandRobot, YBTFlatArmStandCfg(), YBTFlatArmStandCfgPPO())
task_registry.register( "ybt_arm_walk", YBTArmWalkRobot, YBTFlatArmWalkCfg(), YBTFlatArmWalkCfgPPO())
