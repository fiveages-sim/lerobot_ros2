# 本工作区环境说明（grasp / ros2-stack）

**唯一开发目录：** `/home/fiveages/lerobot_ros2`（本仓库）。  
其它机器上的同名 clone 不作为本功能的环境来源；文档与脚本只引用本路径。

## 两仓比对结论（2026-09）

与另一份 checkout 对照后：

| 项 | 结论 |
|----|------|
| 父仓 `origin` | 相同（`fiveages-sim/lerobot_ros2`） |
| 父仓 HEAD | **相同** `d3221de`（本仓分支 `feature/sim-grasp-datagen`） |
| `robot_action_composer` | 基线 commit **相同**；**本仓有未提交的 grasp-generation 开发**，另一份没有 → **无冲突，以本仓为准** |
| `ros2_robot_interface` | 本仓分支更新（含更多 main 提交）；另一份停在更旧 main → **另一份是本仓祖先，无分叉冲突** |
| 新功能代码 | 仅在本仓（`grasp_generation/`、`grasp-generation` CLI、HUG submodule、文档） |

**无需合并冲突处理**；之后只在本仓提交 / 推送。

## 一键配环境

```bash
cd /home/fiveages/lerobot_ros2
bash scripts/setup_grasp_dev_env.sh
source .venv/bin/activate
```

`.venv`：Python **3.12** + `--system-site-packages`（用系统 `rclpy`）。  
`submodules/hug/.venv`：Python **3.10**，仅 HUG 推理，不要混用。

## 日常启动

```bash
cd /home/fiveages/lerobot_ros2
source .venv/bin/activate          # 自动 source jazzy + ros2_ws + fa_w2_ws

cd examples/IsaacSim
ros2-stack launch --robot fiveages_w2

# 另开终端同样 activate 后：
grasp-generation serve --robot fiveages_w2 --workspace .
```

## 禁止

- 用系统 `pip install`（会 `externally-managed-environment`）
- 在 `submodules/hug/.venv` 里装 `ros2-stack` / `grasp-generation`
- 把本功能的环境依赖指到其它 checkout 的 `.venv`
- 在本仓 `.venv` 里装 **NumPy 2.x**（系统 `cv2` 按 1.x 编译，会 `multiarray failed to import`；保持 `numpy>=1.26,<2`）
