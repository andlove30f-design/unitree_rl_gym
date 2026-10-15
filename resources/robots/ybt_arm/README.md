# ybt_arm — 带臂四足资产

YBT 四足 + RM65-B 六轴机械臂，机械臂焊在四足躯干（`trunk`）顶部，合并成**一个** URDF。

```
resources/robots/ybt_arm/
├── urdf/ybt_arm.urdf          # 合并结果，25 links / 24 joints，18 个可动关节
├── tools/build_ybt_arm.py     # 生成脚本，改安装位姿后重跑即可
└── README.md
```

两个源模型没有被复制，都还是原地引用：

* 四足：`resources/robots/ybt/`
* 机械臂：`resources/robots/RM65-B/`

重新生成：

```bash
python resources/robots/ybt_arm/tools/build_ybt_arm.py
```

在 Isaac Gym 里看：

```bash
conda activate unitree-rl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python legged_gym/scripts/view_ybt.py --urdf resources/robots/ybt_arm/urdf/ybt_arm.urdf
```

`view_ybt.py` 会先打印 mesh 检查表、关节表、质量表，不写 `--headless` 就弹 viewer
窗口（`--dynamic` 让躯干自由落体）。

训练任务已经建好了，任务名 **`ybt_arm`**，机械臂冻结当配重，折叠姿态见下一节。
用法和参数见 [doc/ybt_training_zh.md](../../../doc/ybt_training_zh.md) 的「带臂版」一节。

## 安装位置是怎么定的

| 依据 | 数值 |
| --- | --- |
| `trunk` 顶面（从 `trunk.dae` 顶点量出来的平台高度） | z = +0.1012 m |
| `base_link` 原点（`base_link.STL` 的 z 范围 0 → +0.1385） | 正好在机械臂的安装底面 |
| 焊接关节 | `trunk` --`arm_mount_joint`--> `arm_base_link`，`xyz="0 0 0.1012" rpy="0 0 0"` |

也就是把机械臂底面直接贴在躯干顶面上、放在背部正中（x = y = 0），并且 `rpy` 全 0，
这样 `joint_1` 的转轴正好竖直向上，机械臂零位是"立正"的。

要挪位置就改 `tools/build_ybt_arm.py` 顶部的 `MOUNT_XYZ` / `MOUNT_RPY` 再重跑脚本，
不要手改 URDF。

机械臂的 link 和 joint 统一加了 `arm_` 前缀，避免和四足重名：

| 原 RM65-B | 合并后 |
| --- | --- |
| `base_link`, `link_1` … `link_6` | `arm_base_link`, `arm_link_1` … `arm_link_6` |
| `joint_1` … `joint_6` | `arm_joint_1` … `arm_joint_6` |

## 零位姿态与尺寸

六个关节全部为 0 时，机械臂竖直向上（位置在 `trunk` 坐标系下）：

| link | z |
| --- | --- |
| `arm_base_link` | +0.1012 |
| `arm_link_1` / `arm_link_2` | +0.3417 |
| `arm_link_3` | +0.5977 |
| `arm_link_4` / `arm_link_5` | +0.8077 |
| `arm_link_6` | +0.9517 |

四足站姿下 `trunk` 原点离地 0.6202 m，所以零位时机械臂末端大约在 1.57 m 高。
整个资产的最低碰撞点和纯四足一样是 −0.6202 m（足底球），机械臂不会伸到脚底下。
生成高度（spawn height）仍是 0.6202 m。

## 训练时的折叠姿态（任务 `ybt_arm`）

零位（立正）虽然在几何上没问题，但第一阶段的训练目标是**先走起来**，机械臂只是配重，
所以 `ybt_arm` 任务把机械臂折起来贴在背部，用它当姿态的**默认角**（冻结时 PD 就按住这个角）：

| joint | 角度 | 说明 |
| --- | --- | --- |
| `arm_joint_1` | 0.0 | 底座偏航，居中 |
| `arm_joint_2` | +1.0 | 肩俯仰，向后折 57° |
| `arm_joint_3` | +2.0 | 肘，继续折 115°（前臂从肘部挂回背部） |
| `arm_joint_4` | 0.0 | 腕滚 |
| `arm_joint_5` | +0.8 | 腕俯仰，把手抬离甲板 |
| `arm_joint_6` | 0.0 | 工具滚 |

这个姿态是**算出来的**，不是试出来的：

```bash
python resources/robots/ybt_arm/tools/folded_pose.py          # 汇报任务用的姿态
python resources/robots/ybt_arm/tools/folded_pose.py --grid   # 重新搜索 q2/q3/q5
python resources/robots/ybt_arm/tools/folded_pose.py 1.0 2.0 0.8   # 试一组角
```

脚本解析本 URDF 的关节链做正向运动学、读 7 个碰撞 STL 的顶点包围盒，角度直接从
`ybt_arm_config.py` 里读，所以不会和任务配置走散。**改角度后重跑它核对最低点和质心。**

零位 vs 折叠（下表都是"机械臂"= `arm_link_1..6` 这 5.311 kg 可动部分；`arm_base_link`
1.862 kg 是固定关节、导入时折进了 `trunk`）：

| | 零位（全 0） | 折叠（上表） |
| --- | --- | --- |
| 机械臂最低碰撞点 z（`trunk` 系） | +0.240（`arm_link_1`）/ +0.301（其余） | **+0.140**（`arm_link_6`，手） |
| 机械臂质心 (x, y, z) | (0.000, 0.003, +0.552) | (**−0.137**, 0.003, **+0.321**) |
| 绕躯干原点二阶矩 Σm·r² | 1.918 kg·m² | **0.742 kg·m²**（降 61%） |
| 静态重力力矩 | 六个关节全是 0 | j2 = **+7.1 N·m**，其余 ≤ 0.4 |

读法：

* 最低点 0.140 m 是**手**，比顶甲板（+0.1012）高 4 cm、比躯干碰撞盒顶面（+0.09）高 5 cm，
  而且 x = −0.285 已经在碰撞盒（x 半宽 0.253）外面，双保险不会插进躯干。
  `arm_link_1` 是肩座，两个姿态下都停在 +0.240，不随 `q2/q3/q5` 动。
* 占用 `x ∈ [−0.285, +0.056]`、`y ∈ [−0.047, +0.068]`，压在背部后半段、横向居中，
  不碰前部传感器区，也不碰腿（髋在躯干侧面、甲板以下）。
* `+q2/+q3/+q5` 是往**后**折；想往前折就把这三个角同时取负，其余数字完全镜像。
* 零位其实是**静平衡**的（质心正好在躯干竖轴上，六个关节重力矩全为 0），所以折叠不是为了
  省力矩，而是为了**降低转动惯量和高度**；代价只是肩关节多了一个 7.1 N·m 的恒定负载
  （3.80 kg 的远端质量、0.19 m 力臂），相对 60 N·m 的上限可以忽略。

## 关节

机械臂部分（速度上限都是 3.14 rad/s）：

| joint | 范围 (rad) | effort (N·m) |
| --- | --- | --- |
| `arm_joint_1` | ±3.106 | 60 |
| `arm_joint_2` | ±2.2689 | 60 |
| `arm_joint_3` | ±2.356 | 30 |
| `arm_joint_4` | ±3.106 | 10 |
| `arm_joint_5` | ±2.234 | 10 |
| `arm_joint_6` | ±6.28 | 10 |

四足部分 12 个关节 + 4 个固定足端关节，完全没动。

## 质量

| 项 | 质量 |
| --- | --- |
| 机械臂 7 个 link 合计 | 7.173 kg |
| 四足（不含机械臂） | 61.669 kg |
| 合计 | **68.842 kg** |

`arm_base_link`（1.862 kg）是固定关节，导入时被折叠进 `trunk`，所以 Isaac Gym 里
`trunk` 是 28.3817 kg。

## 网格：为什么视觉用 dae、碰撞用 STL

这是整个资产唯一需要解释的地方，结论是**实测**出来的，不是猜的。

Isaac Gym 用 `asset_options.flip_visual_attachments = True`（本仓库默认，四足需要它）。
把同一套机械臂几何分别用 STL 和 `.dae` 各渲染一次、和"原始几何的渲染结果"逐像素对比，
再把机械臂六个关节焊死让它自由落体来量碰撞体，结果：

| 网格格式 | 视觉（flip=True） | 碰撞体 |
| --- | --- | --- |
| `*.STL` | 散架，被整体转错 | **正确** |
| `*.dae`（`<up_axis>Z_UP</up_axis>`） | **和原始几何完全一致** | 被转错 |

两个信息合起来解释得通：assimp 读 COLLADA 时会按文件声明的 `up_axis` 做轴向转换，而
读 STL 时文件里没有这个信息、不做转换；`flip_visual_attachments` 只作用在视觉网格上，
不能把碰撞体的这次转换撤销。

碰撞体错位的量也量到了：焊死的机械臂自由落体后，STL 碰撞停在 z ≈ 0.005 m，
dae 碰撞停在 z ≈ 0.057 m，差的 52 mm 正好是底座盘的半宽 0.055 m。

所以：

* **视觉** → `RM65-B/meshes/visual/*.dae`（由 `RM65-B/tools/stl_to_dae.py` 从 STL 转出，
  只加 `<up_axis>Z_UP</up_axis>` 元数据，顶点数据一名不差）
* **碰撞** → `RM65-B/meshes/*.STL`（厂商原文件）

四足那边一直没暴露这个问题，是因为它的碰撞几何**全是 box / cylinder / sphere 基本体**，
一个网格碰撞都没有；它的 dae 只当视觉用，而 dae + flip=True 恰好是正确的组合。

想重新生成视觉网格：

```bash
python resources/robots/RM65-B/tools/stl_to_dae.py \
    resources/robots/RM65-B/meshes \
    resources/robots/RM65-B/meshes/visual
```

碰撞网格**保持原样没换**。原本担心 7 个高精度 STL 凸包的开销，实测结果是带臂版比无臂版慢
约 1.5 倍（4096 环境：97k vs 145k steps/s，差距都在 collection 阶段），训练预算能接受，
所以没有退化成 box / capsule。

## 还没做的

* 机械臂和躯干、机械臂自身的自碰撞没有专门调过（`self_collisions` 沿用仓库默认）。
* 机械臂只当配重，没有任何属于机械臂自己的动作或奖励。想解冻就把
  `ybt_arm_config.py` 里的 `control.arm_action_scale` 改成腿的值（0.25），
  再给机械臂加奖励项；其余部分不用动。
