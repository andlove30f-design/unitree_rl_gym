<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>This is a repository for reinforcement learning implementation based on the YBT quadruped robot (Isaac Gym training only).</strong>
</p>

---

## 📦 Installation and Configuration

Please refer to [setup.md](/doc/setup_en.md) for installation and configuration steps.

## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play`

- **Train**: Use the Gym simulation environment to let the robot interact with the environment and find a policy that maximizes the designed rewards. Real-time visualization during training is not recommended to avoid reduced efficiency.
- **Play**: Use the Play command to verify the trained policy and ensure it meets expectations.

## 🛠️ User Guide

### 1. Training

Run the following command to start training:

```bash
# Plain quadruped
python legged_gym/scripts/train.py --task=ybt

# Armed quadruped (RM65-B arm folded and frozen as payload; walking comes first)
python legged_gym/scripts/train.py --task=ybt_arm

# Armed quadruped standing still while the arm works (legs learn to hold the trunk)
python legged_gym/scripts/train.py --task=ybt_arm_stand

# Armed quadruped walking while the arm works (the hardest of the four)
python legged_gym/scripts/train.py --task=ybt_arm_walk
```

#### ⚙️ Parameter Description
- `--task`: `ybt` (plain quadruped), `ybt_arm` (quadruped carrying an RM65-B arm),
  `ybt_arm_stand` (the same robot, standing: zero velocity commands, the arm swings on a
  script and the legs balance against it) or `ybt_arm_walk` (the scripted swing while the
  legs keep following the walking velocity commands).
- `--headless`: Defaults to starting with a graphical interface; set to true for headless mode (higher efficiency).
- `--resume`: Resume training from a checkpoint in the logs.
- `--experiment_name`: Name of the experiment to run/load.
- `--run_name`: Name of the run to execute/load.
- `--load_run`: Name of the run to load; defaults to the latest run.
- `--checkpoint`: Checkpoint number to load; defaults to the latest file.
- `--num_envs`: Number of environments for parallel training.
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.
- `--sim_device`: Simulation computation device; specify CPU as `--sim_device=cpu`.
- `--rl_device`: Reinforcement learning computation device; specify CPU as `--rl_device=cpu`.

Play/inspection only:
- `--cmd_vx` / `--cmd_vy` / `--cmd_wz`: Hold a velocity command fixed instead of re-randomizing it.
- `--cmd_heading`: Hold a target heading fixed, in degrees (the robot turns to face it). Mutually exclusive with `--cmd_wz`.

**Default Training Result Directory**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

To visualize the training results in Gym, run the following command:

```bash
python legged_gym/scripts/play.py --task=ybt
python legged_gym/scripts/play.py --task=ybt_arm
python legged_gym/scripts/play.py --task=ybt_arm_stand
python legged_gym/scripts/play.py --task=ybt_arm_walk
```

**Description**:

- Play’s parameters are the same as Train’s.
- By default, it loads the latest model from the experiment folder’s last run.
- You can specify other models using `load_run` and `checkpoint`.

#### 💾 Export Network

Play exports the Actor network, saving it in `logs/{experiment_name}/exported/policies`:
- Standard networks (MLP) are exported as `policy_1.pt`.
- RNN networks are exported as `policy_lstm_1.pt`.

## 🔧 YBT Utilities

- `legged_gym/scripts/view_ybt.py`: Inspect the YBT URDF and meshes in the Isaac Gym viewer without training.
- `legged_gym/scripts/check_ybt.py`: Optional headless smoke test (physics, PPO, checkpoint reload, TorchScript export). Works for all four tasks; pick one with `--task`. On the scripted-arm tasks (`ybt_arm_stand`, `ybt_arm_walk`) it also measures how far the arm lags its reference and how far the swing tilts the trunk.
- `resources/robots/ybt_arm/tools/folded_pose.py`: Check the armed task's folded arm pose -- lowest collision point, centre of mass, inertia and static torques. `--sweep` does the same along the scripted-arm tasks' working swing.
- `tests/test_ybt.py` / `tests/test_ybt_arm.py` / `tests/test_ybt_arm_stand.py` / `tests/test_ybt_arm_walk.py`: CPU regression tests for the asset, config, default stand and reset logic, plus the armed task's freeze and folded pose, the scripted reference and curriculum (shared by both scripted-arm tasks), and the walking task's commands, class hierarchy and gates.

```bash
python -m unittest discover -s tests -p "test_ybt*.py" -v
```

All YBT tasks, their reward design, their default training parameters and the armed versions are documented in [doc/ybt_training_zh.md](doc/ybt_training_zh.md) (Chinese).

## 🎉 Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): The foundation for training and running codes.
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.

---

## 🔖 License

This project is licensed under the [BSD 3-Clause License](./LICENSE):
1. The original copyright notice must be retained.
2. The project name or organization name may not be used for promotion.
3. Any modifications must be disclosed.

For details, please read the full [LICENSE file](./LICENSE).
