<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>这是一个基于 YBT 四足机器人实现强化学习的仓库（仅包含 Isaac Gym 训练）。</strong> 🚪🎮
</p>

---

## 📦 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py --task=ybt
```

#### ⚙️  参数说明
- `--task`: 当前仓库只注册了 `ybt` 这一个任务。
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`

Play/查看专用：
- `--cmd_vx` / `--cmd_vy` / `--cmd_wz`: 固定速度命令，不再每 10 秒重新随机
- `--cmd_heading`: 固定目标朝向（单位：度），机器人转过去并保持；与 `--cmd_wz` 互斥

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py --task=ybt
```

**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`

## 🔧 YBT 工具

- `legged_gym/scripts/view_ybt.py`: 不训练，直接在 Isaac Gym 查看器中检查 YBT 的 URDF 和网格。
- `legged_gym/scripts/check_ybt.py`: 可选的无界面短测（物理、PPO、检查点重载、TorchScript 导出）。
- `tests/test_ybt.py`: 资源、配置、默认站姿和随机复位的 CPU 回归测试。

```bash
python -m unittest discover -s tests -p test_ybt.py -v
```

YBT 任务的奖励设计和默认训练参数详见 [doc/ybt_training_zh.md](doc/ybt_training_zh.md)。

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。

---

## 🔖  许可证

本项目根据 [BSD 3-Clause License](./LICENSE) 授权：
1. 必须保留原始版权声明。
2. 禁止以项目名或组织名作举。
3. 声明所有修改内容。

详情请阅读完整 [LICENSE 文件](./LICENSE)。
