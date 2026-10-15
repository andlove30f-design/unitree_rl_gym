# YBT 四足：Isaac Gym 训练接入说明

## 1. 已完成的内容

YBT 已注册为独立任务 `ybt`，复用仓库的四足观测、PD 力矩控制和 PPO 训练流程。
当前范围仅包含 **Isaac Gym 平地训练与 Play/策略导出**，不包含 YBT 的 MuJoCo 或实机部署。

新增文件：

| 文件 | 用途 |
| --- | --- |
| `legged_gym/envs/ybt/__init__.py` | YBT 环境包 |
| `legged_gym/envs/ybt/ybt_config.py` | 模型路径、默认站姿、控制参数、奖励、训练参数 |
| `legged_gym/envs/ybt/ybt_env.py` | 模型预检查、四足识别检查、限位内随机复位 |
| `legged_gym/scripts/check_ybt.py` | 可选的无界面短测，验证物理、PPO、保存/重载和导出 |
| `tests/test_ybt.py` | 配置、资源、惯量、默认站姿、随机复位的回归测试 |
| `doc/ybt_training_zh.md` | 本说明 |

修改文件：

- `legged_gym/envs/__init__.py`：只导入并注册 `ybt`（Go2/G1/H1/H1_2 已从本仓库删除）。
- `README.md` / `README_zh.md`：改为只描述 YBT 的训练、Play 和工具入口。
- `resources/robots/ybt/urdf/ybt.urdf`：仅将一处 `orange` 颜色从 0..255 归一化到 0..1；没有修改几何、质量、惯量、关节名或限位。

使用用户已添加的资源（五个网格没有重新生成或转换）：

```text
resources/robots/ybt/
├── urdf/ybt.urdf
└── meshes/
    ├── trunk.dae
    ├── hip.dae
    ├── thigh.dae
    ├── thigh_mirror.dae
    └── calf.dae
```

## 2. 为什么不能直接复制 Go2 配置

- YBT 模型总质量约 **61.67 kg**，控制增益不能简单沿用 Go2 的 `20 / 0.5`。
- YBT 大腿、膝关节轴为 `0 -1 0`；站姿使用 **大腿 -0.8 rad、膝关节 +1.6 rad**，而不是 Go2 的正大腿、负膝角。
- 膝关节的 URDF 限位约为 `[1.0297, 2.7227]` rad。原环境以默认角乘 `0.5..1.5` 复位时，可能得到 `0.8` rad，越过下限。YBT 改为默认角加 **±0.05 rad** 随机扰动，再夹到软限位内，其他机器人的复位逻辑不变。
- YBT 的机身名是 `trunk`，跌倒接触终止条件必须匹配这个名称；保留四个 `*_foot` 刚体用于接触和足端奖励。
- 默认角按 **关节名称**匹配。动作顺序以 Isaac Gym 加载后的 `dof_names` 为准，不由配置字典的排列决定。本机实测顺序是 FL、FR、RL、RR，每条腿 hip、thigh、calf。

控制初值：hip `Kp=100 / Kd=3`，thigh/calf `Kp=150 / Kd=4`，动作缩放 `0.25`。
这些是已跑通短测的**仿真起始参数**，并非实机增益或已经优化到最优的控制参数。
出生机身高度 `0.55 m`，奖励目标高度 `0.50 m`。

针对首轮训练暴露出的**走路拖沓/迈不开脚、走几秒就歪**，做了如下调整：

- `heading_command = True`：恢复朝向闭环。基类（go2 等）默认就开，原先只追踪偏航**速率**，残留的小偏航率几乎不受惩罚却会累积成明显偏航。注意：开启后偏航由"朝向误差"闭环生成。`play.py` 为此提供两种互斥的转弯方式：`--cmd_heading=<度>` 指定目标朝向（机器人转过去并保持），`--cmd_wz=<rad/s>` 指定固定转速（此时会临时关掉朝向环，否则该值每一步都被覆盖）。只给 `--cmd_vx/--cmd_vy` 时目标朝向锁在 0，机器人保持出生朝向走直线。
- `feet_clearance` 奖励（`YBTRobot._reward_feet_clearance`）：摆动腿足端高度落在 `feet_clearance_target = 0.05 m` 与 `feet_clearance_cap = 0.09 m` 之间时给满分，**超出 cap 后按同样斜率在一倍跨度内衰减回 0**（0.13 m 以上完全不给分），且整体**钳制在 ≥0**。拖地的脚得 0 分、抬起的脚得正分，所以策略没有"索性不抬腿"的动机；抬过头也没有额外收益，所以也没有"越抬越高"的动机。权重 `feet_clearance = 0.3`：该项是速度追踪的**从属项**——初版取 1.0 时它一度占到总奖励的约 85%。
  早期版本在 cap 处封顶后一直保持满分（没有衰减段），于是没有任何力量把过高的脚压回来：实测空中脚有 95% 超过 cap、摆动腿高度中位数 0.157 m（cap 为 0.09 m），机器人高抬腿踏步，站住不动时也一样。改成"带"以后，把同一个旧步态分别按两种形状打分，旧形状 0.739/s、新形状仅 0.056/s——**丧失约 0.68/s，相当于总奖励的 36%**，梯度方向随之反转。
- 其余奖励保持基类默认：`tracking_ang_vel = 0.5`、`action_rate = -0.01`。`feet_air_time` 由基类的 `1.0` 改为 **`0.0`**（关闭）：基类阈值写死 0.5 s，而本机实测摆动腿腾空时间仅约 0.15 s，于是每次落地都扣分，唯一能减少扣分的办法是把腿抬得更高——方向上和高抬腿是同一个推力。实测它只占总奖励的 0.4%（−0.007/s），量级上无关紧要，但方向有害，故直接归零；`feet_clearance` 已以正确的符号覆盖了同一件事。

以上改动了奖励函数，`logs/flat_ybt/` 下的旧检查点需重新训练。

## 3. 默认训练参数

| 参数 | 默认值 |
| --- | --- |
| 并行环境 | 12288 |
| PPO 迭代上限 | 1500 |
| 每次迭代每环境采样步数 | 24 |
| 动作维度 / 观测维度 | 12 / 48 |
| 物理步长 | 0.005 s（200 Hz） |
| 控制降采样 | 4（策略 50 Hz） |
| 单个 episode 上限 | 20 s（提前跌倒会复位） |
| 模型保存间隔 | 50 次迭代，结束也会保存 |
| 实验目录 | `logs/flat_ybt/` |
| 地形 | 平地，无地形课程 |

48 维观测依次为：机身线速度 3、机身角速度 3、重力方向 3、速度命令 3、关节位置偏差 12、关节速度 12、上一动作 12。当前不包含步态相位。
默认每迭代采集 `12288 × 24 = 294,912` 条转换；1500 次迭代共 `442,368,000` 条。
**1500 是训练预算，不保证固定次数后就能获得理想步态。** 用 Play 检查站立、前后移动、侧移、转向与抗扰表现后，再决定是否继续训练或调整奖励。

## 4. 怎么启动

在仓库根目录，使用已有 `unitree-rl` 环境：

```bash
conda activate unitree-rl
cd /home/yh/unitree_rl_gym

# 正式训练：默认 12288 环境、1500 次迭代
python legged_gym/scripts/train.py --task=ybt --headless

# 显存紧张时，可以减小并行数量
python legged_gym/scripts/train.py --task=ybt --num_envs=4096 --headless

# 查看正式训练结果并导出 actor
python legged_gym/scripts/play.py --task=ybt --num_envs=16
```

检查点保存在 `logs/flat_ybt/<日期时间>_<run_name>/model_<iteration>.pt`。
Play 的导出文件为 `logs/flat_ybt/exported/policies/policy_1.pt`。
新任务默认从头训练，不会自动加载任何预训练模型。

继续某次训练可显式指定 `--resume --load_run=<运行文件夹名> --checkpoint=<编号>`。
既有训练入口和 Play 不需要为 YBT 改动。

## 5. 本次验证结果

在本机 `unitree-rl` 环境和 RTX 4070 上已完成：

- 9 项回归测试全部通过，包括资源非空/可解析、正定惯量、默认角限位、站姿足端离地和随机复位。
- 标准训练入口：64 个并行环境、5 次 PPO 迭代，成功保存检查点。
- 完整检查脚本：64 个环境，100 步物理检查、3 次 PPO 迭代、检查权重更新、检查点重载、TorchScript 导出与输出一致性、50 步策略推理运行，全部通过。
- 实际识别到 12 个关节、4 个足端，观测形状为 `(64, 48)`，检查范围内没有非有限状态或超出 URDF 力矩限值的施加力矩。

按用户要求，确认能训练后已经停止测试，**未跑默认 4096 环境的压力测试，也未完成 1500 次完整训练或验证收敛步态**。
短测产物放在独立的 `logs/ybt_smoke/` 下，不会被正式 `flat_ybt` 的 Play 自动选中。

以下是以后需要时才运行的可选检查，不是启动训练的必要步骤：

```bash
python -m unittest discover -s tests -p test_ybt.py -v
python legged_gym/scripts/check_ybt.py --num_envs=64 --max_iterations=3
```

如果直接调用环境而不是先激活 Conda，需要把环境的 `bin` 加入 PATH，以便找到 Ninja；若遇到 `libpython3.8.so` 加载错误，可在已激活的环境中设置：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

检查脚本与正式训练均需安装现有 Isaac Gym、PyTorch 和 rsl_rl。CPU 回归测试本身不创建 GPU 仿真。
