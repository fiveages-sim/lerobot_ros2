# 资产 / 运控侧对接（第一期 W2 + LinkerHand O7）

> 未经同意，自动化流程不改 FaSim-Isaac / fa_w2_ws / USD 原始文件。

## A. 头相机（已对齐 · 2026-09）

| 项 | 值 |
|----|-----|
| usda | `humanoid/FiveAges/Gen2/W2/env/dexhand_o7_new.usda` |
| camera_prim | `/World/FiveAges_W2/Head_V1/head_link2/head_camera/head_camera` |
| rgb | `/head_camera/rgb` |
| depth | `/head_camera/depth`（`32FC1` 米） |
| camera_info | `/head_camera/camera_info` |
| resolution | 640×480，fx≈fy |
| physics | PhysX（`dexhand_o7_new`） |

资产侧已在头部创建光学 prim，测试 env 临时 graph 已验证 ROS2 通信、RGB/Depth 正常。

父仓编排侧已补 `examples/IsaacSim/robots/FiveAges_W2/lerobot_config.py`：

```python
depth_camera_name="head_camera",
depth_info_topic="/head_camera/camera_info",
# head_camera.depth_topic_name="/head_camera/depth"
```

第一期测试场景：W2 + O7 + 桌上可被头相机看见的一对物体 → 测 HUG / MANO。

## B. fa_w2_ws / 运控（先对齐，暂不改）

O7 关节顺序（`linkerhand_description` `o7.side.yaml`）：

```text
{side}_hand_thumb_joint1/2/3
{side}_hand_index_joint
{side}_hand_middle_joint
{side}_hand_ring_joint
{side}_hand_pinky_joint
```

下发：`ROS2RobotInterface.send_left/right_hand_joint_positions` → `/left|right_hand_controller/target_joint_position`。

Launch：`type:=linkerhand_o7`。父仓 `robot.yaml` motion 已对齐为 `type: linkerhand_o7`。

## C. 父仓 / composer 状态

| 项 | 状态 |
|----|------|
| 分支 | 父仓 `feature/sim-grasp-datagen`；composer `feature/dex-grasp-generator` |
| `submodules/hug` | 本地 gitlink，**不 push**；uv `.venv` + MANO + 权重冒烟已过 |
| `lerobot_config.py` depth | **已补** |
| RGB-D → HUG + Viser + 跟腕冒烟 | **已通**；腕→EE 有平移偏差 → **2A.1** |
| 臂 EE 对齐（非手指） | **进行中** — [`SIM_GRASP_EE_ALIGN.md`](SIM_GRASP_EE_ALIGN.md) |
| MANO→O7 指关节 retarget / skill | **对齐后**（不改 HUG 权重） |

## D. 怎么测（详细步骤）

完整操作：[`SIM_GRASP_HUG_TEST_GUIDE.md`](SIM_GRASP_HUG_TEST_GUIDE.md)。  
联调结论与后续路线图：[`SIM_GRASP_MILESTONE_AND_ROADMAP.md`](SIM_GRASP_MILESTONE_AND_ROADMAP.md)。

```bash
cd submodules/hug && source .venv/bin/activate
export PYTHONPATH=/home/fiveages/lerobot_ros2/submodules/robot_action_composer:${PYTHONPATH}
bash scripts/smoke_w2_head_hug.sh --u-norm 0.48 --v-norm 0.42
# 看手：DATA=data/w2_head_capture bash scripts/run_app.sh → 浏览器点物体
```
