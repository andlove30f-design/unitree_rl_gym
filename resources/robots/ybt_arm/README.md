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
窗口（`--dynamic` 让躯干自由落体）。目前还没写任何训练代码。

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

## 还没做的

* 机械臂的碰撞网格还是原始高精度 STL，PhysX 导入时会取凸包（每个 link 一个）；
  真开始训练前可能想换成简化体。
* 电机参数（`stiffness` / `damping` / `effort`）还没写进任何配置——目前只导入了 URDF
  里写的 effort / velocity 上限。
* 机械臂和躯干、机械臂自身的自碰撞没有专门调过。
