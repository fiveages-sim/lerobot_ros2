#!/usr/bin/env bash
# 采集 / 回放末端轨迹的便捷入口。
#
# 这个脚本只做一件事：把 ROS 2 + zenoh 环境准备好，然后转发给 ee_traj.py。
# 环境变量与 Isaac(Play) 一致，缺了会看不到任何话题。
#
# 用法:
#   ./ee_traj.sh check
#   ./ee_traj.sh record -o ~/ee_traj/pick_01.npz --duration 30
#   ./ee_traj.sh inspect ~/ee_traj/pick_01.npz
#   ./ee_traj.sh replay ~/ee_traj/pick_01.npz
#
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- ROS 环境（与 Isaac Play 时的运控节点一致）----
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ZENOH_CFG="${ZENOH_SESSION_CONFIG_URI:-/home/fiveages/fa_w2_ws/zenoh/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5}"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[ERROR] 找不到 $ROS_SETUP" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ROS_SETUP"

# 自定义消息（arms_ros2_control_msgs 等）在工作空间里；没 source 会读不到 WBC 状态
WS_SETUP="${ROS_WS_SETUP:-/home/fiveages/ros2_ws/install/setup.bash}"
if [[ -f "$WS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$WS_SETUP"
else
  echo "[warn] 找不到 $WS_SETUP；WBC 状态检查会跳过" >&2
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
if [[ -f "$ZENOH_CFG" ]]; then
  export ZENOH_SESSION_CONFIG_URI="$ZENOH_CFG"
else
  echo "[warn] zenoh 配置不存在: $ZENOH_CFG（仍继续，但可能发现不到话题）" >&2
fi

# ---- Python：需要 rclpy。优先用 lerobot 的 conda 环境 ----
PYTHON="${EE_TRAJ_PYTHON:-/home/fiveages/miniconda3/envs/lerobot-ros2/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
if ! "$PYTHON" -c "import rclpy" >/dev/null 2>&1; then
  echo "[ERROR] $PYTHON 里没有 rclpy。" >&2
  echo "        设 EE_TRAJ_PYTHON 指向一个 source 过 ROS 的 python。" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/ee_traj.py" "$@"