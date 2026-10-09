# 本工作区环境说明（grasp / ros2-stack）

**唯一开发目录：** 本仓库 clone 根目录。文档与脚本不再写死某一台机器的绝对路径。

抓取与运控共用 **父仓 `.venv`（Python 3.12 + 系统 rclpy）**。HUG 推理单独用 **`submodules/hug/.venv`（Python 3.10）**，不要混装 CLI。

## 一键配环境（与 motion 同一条链路）

```bash
git clone --recursive <this-repo>
cd lerobot_ros2
./init.sh all-motion          # 子模块 + .venv + interface/composer + viser + wuji-sdk
./init.sh ros2-workspace      # 激活环境时自动 source ROS2 工作空间
# 需要头相机 HUG 推理时：
./init.sh hug-env
source .venv/bin/activate
```

`scripts/setup_grasp_dev_env.sh` 仅是 `./init.sh all-motion` 的别名，新流程请直接用 `init.sh`。

## 日常启动

```bash
source .venv/bin/activate

cd examples/IsaacSim
ros2-stack launch --robot fiveages_w2

# 另开终端同样 activate 后：
grasp-generation              # 交互；或 grasp-generation serve --robot fiveages_w2 --workspace .
```

## 禁止

- 用系统 `pip install`（会 `externally-managed-environment`）
- 在 `submodules/hug/.venv` 里装 `ros2-stack` / `grasp-generation`
- 把环境依赖指到其它 checkout 的 `.venv`
- 在本仓 `.venv` 里为了 lerobot 强行装 **NumPy 2.x** 后再跑依赖系统 `cv2` 的抓取（`./init.sh all` / `install-lerobot` 会装 numpy 2；纯抓取用 `all-motion`）
