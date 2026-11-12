# G1D 末端轨迹工具（EE-only）

用**末端位姿 + 夹爪**作为唯一的动作接口来采集与验证数据，绕开 Isaac 视窗操作。

## 为什么走这条路

训练目标就是「输入相机 → 输出双臂末端位姿 + 夹爪开合」，末端位姿再交给运控
(OCS2 全身控制) 做 IK/MPC。既然如此：

- 验证数据质量，不需要回放 32 个关节角、不需要摆 Isaac 视窗、不需要管底盘开合。
- 只要把采集到的末端轨迹重新喂给运控，看它能不能跟住，就是最贴近部署的自检。
- 运控在轮式模式下所有末端目标都在 **world** 绝对系，所以底盘动没动不影响数据有效性。

## 文件

| 文件 | 作用 |
|---|---|
| `ee_traj.py` | 采集 / 检查 / 截取 / 回放 |
| `ee_traj.sh` | 环境包装（ROS + zenoh + 正确的 python） |

## 两套位姿 + 一套关节角：observation / action

| | 话题 | 含义 | 用途 |
|---|---|---|---|
| 实测 | `/*_current_pose` | 控制器**跟出来**的位姿 | 训练 observation |
| 指令 | `/*_current_target` | 当时**下发**的目标 | 训练 action 标签；回放复现 |
| 关节角 | `/joint_states` 里的 18 个运控关节 | 实测关节位置+速度 | **ACT 类方法复刻流程** |

为什么要同时存关节角：末端目标对 `.info` 的耦合/权重很敏感，改一次权重整条末端
轨迹就变了；而关节角不会。ACT 想稳定复刻任务流程，就得有关节角这条独立的标签。

拿实测当 action 会把控制器的跟踪误差学进策略里。文件里两套位姿都存，
`inspect` 会打印差值——差值大说明那段运控本来就追不上，不适合直接训练。

采集的 18 个关节（顺序固定，`inspect` 会打印）：
`LZ_mt_Joint, LZ_it_Joint, waist_pitch, waist_yaw, left×7, right×7`
前两个是**升降柱 prismatic（0~0.21m）**，不是旋转关节。

## 底盘：全程 LOCK，不需要移动

交付要求整场底盘不动、且目标平台可能不是轮式，所以编排第 0 步就 `BASE_LOCK`，
第 8 步原来的 `BASE_UNLOCK` 已 mute（2026-11-05）。

这个改动是验证过的，不是猜的。用运控实际使用的 planning URDF 做 IK：
`hole_3` 所在的 `test_tube_rack` 在 base 系 `前 0.613 / y=+0.082 / 高 0.796m`，
左 tcp 带「夹爪垂直向下」姿态约束仍有 **169/200 个解**，其中存在
**升降柱完全不抬（`LZ_mt=LZ_it=0`）+ 腰部只弯 26~62°** 的自然解。
所以「够不到要靠底盘挪」这个前提在几何上不成立。

要恢复原行为，把第 8 步改回 `commands: [BASE_UNLOCK, RIGHT_ARM_DISABLE]`。

## 快速开始

```bash
cd examples/IsaacSim/robots/Unitree/G1D/tools

# 1) 前置检查（仿真要在 Play，运控要起着）
./ee_traj.sh check

# 2) 采集：编排在跑，本脚本只旁听，Ctrl+C 结束
./ee_traj.sh record -o ~/ee_traj/pick_01.npz

# 3) 只读检查（不需要 ROS）
./ee_traj.sh inspect ~/ee_traj/pick_01.npz

# 4) 截一段验证（比如只测胸前交接）
./ee_traj.sh slice ~/ee_traj/pick_01.npz --start-sec 12 --end-sec 15

# 5) 回放给运控，看跟踪误差
./ee_traj.sh replay ~/ee_traj/pick_01.npz --approach-duration 3.0
```

## `check` 会报什么

- 左右末端位姿话题是否在发、`frame_id` 是什么
  （**实测是 `base_footprint`**，不是 world；脚本如实记录并原样回发）
- WBC 状态：`base`（LOCKED/UNLOCKED）、`left/right_arm`（ENABLED/DISABLED）、
  `bimanual`（COUPLED/INDEPENDENT）

三种会导致「发了没反应」的情况，`check` 会提前指出：

1. **某臂 DISABLED** → 该臂的 `/left_target`、`/right_target` 被控制器直接丢弃
   （`block_*_external_ee_`）。需要先 `LEFT_ARM_ENABLE` / `RIGHT_ARM_ENABLE`。
2. **BIMANUAL_COUPLED** → `/right_target` 被忽略（左臂 leader 优先）。
   回放脚本默认先发 `ARMS_INDEPENDENT`。
3. **frame 不一致** → 回放原样回发同一 frame，控制器自己做 TF；但如果话题
   frame 和 `base_frame` 都对不上就会整体偏移。

> QoS 坑：`/ocs2_wbc_controller/current_state` 是 **transient_local**。
> 用默认 VOLATILE QoS 订阅会**一条消息都收不到**（`ros2 topic echo` 能收到，
> 因为它自动匹配）。脚本已用 `TRANSIENT_LOCAL + RELIABLE` 订阅。

## `replay` 的流程

1. （默认）发 `ARMS_INDEPENDENT`、`BASE_LOCK`
2. 用 `/dual_target/stamped` 走一条 2 点 moveL 到轨迹起点（`--approach-duration`，
   默认 3s）——避免第一帧从当前位姿直接跳过去触发大速度
3. 按原采样时刻流式发布 `/left_target` + `/right_target`（`Pose`，无 frame，
   控制器按 `base_frame_`=`world` 解释）；夹爪只在命令变化时发一次
4. 保持末帧，打印「指令 vs 实测」的位置/姿态误差

判定参考：**p95 位置误差 < 20 mm 且 max < 60 mm**，说明运控能跟住这段轨迹。

常用参数：

| 参数 | 说明 |
|---|---|
| `--source {auto,cmd,meas}` | 回放哪套位姿。`auto`（默认）有指令就用指令 |
| `--rate-scale 0.5` | 半速回放，看是不是速度问题 |
| `--start/--end` | 按帧截取 |
| `--no-lock-base` | 不锁底盘（默认锁：数据在 world 绝对系，锁住最接近部署） |
| `--no-measure` | 不统计跟踪误差 |
| `--extra-mode RIGHT_ARM_ENABLE` | 额外下发一条模式命令 |

## 与 `LEFT_ARM_DISABLE` 的关系

编排在右手抓取阶段会 `LEFT_ARM_DISABLE`，那段左手没有指令（保持位姿）。
回放时左手会被拉到那个保持位姿——通常没问题（本来就是它当时的位置），
但如果截取的段落跨了 enable/disable 边界，用 `slice` 掐掉更干净。

## 环境要求

`ee_traj.sh` 会自动 source ROS 并设置 zenoh。手动跑 `ee_traj.py` 时需要：

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ZENOH_SESSION_CONFIG_URI=/home/fiveages/fa_w2_ws/zenoh/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5
```

Python 需要带 `rclpy`（默认用 `/home/fiveages/miniconda3/envs/lerobot-ros2/bin/python`，
可用 `EE_TRAJ_PYTHON` 覆盖）。`inspect` / `slice` 不需要 ROS。
