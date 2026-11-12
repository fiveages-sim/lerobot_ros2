#!/usr/bin/env bash
# 重启仿真后的自检：确认模型/身体模式/arm enable 三件事都对。
#
# 用法（仿真已经在 Play）：
#   ./verify_after_restart.sh
#
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
[[ -f /home/fiveages/ros2_ws/install/setup.bash ]] && source /home/fiveages/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export ZENOH_SESSION_CONFIG_URI="${ZENOH_SESSION_CONFIG_URI:-/home/fiveages/fa_w2_ws/zenoh/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5}"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
PY="${VERIFY_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

echo "=================================================================="
echo "1) OCS2 模型（必须是 mobile_base_tcp）"
echo "=================================================================="
INFO=$(timeout 10 ros2 param get /ocs2_wbc_controller info_file_name 2>/dev/null | sed 's/.*: //')
echo "   info_file_name = ${INFO:-<读不到>}"
if [[ "$INFO" == fixed_base_tcp ]]; then
  echo "   [FAIL] 仍是 fixed_base_tcp —— 规划里没有底盘自由度，arm enable 会卡死。"
  echo "          请确认 fiveages.usda 的 inputs:infoFile 为空，然后重新 Play。"
elif [[ "$INFO" == mobile_base_tcp ]]; then
  echo "   [OK]"
else
  echo "   [WARN] 未预期的值"
fi

echo
echo "=================================================================="
echo "2) WBC 能力（basePlan 应为 1）"
echo "=================================================================="
timeout 10 ros2 param get /ocs2_wbc_controller fsm_default_humanoid_mode 2>/dev/null | sed 's/^/   /'

echo
echo "=================================================================="
echo "3) 状态 + 末端位姿 + 关节角"
echo "=================================================================="
cd "$SCRIPT_DIR" && ./ee_traj.sh check 2>&1 | grep -v "ThreadId\|zenoh::"

echo
echo "=================================================================="
echo "4) DISABLE -> ENABLE 往返（最关键）"
echo "=================================================================="
timeout 90 "$PY" "$SCRIPT_DIR/delay_probe.py" --delays 0.5 --enable-timeout 15 2>&1 \
  | grep -v "ThreadId\|zenoh::"

echo
echo "=================================================================="
echo "完成。上面第 4 步出现 OK 就说明 arm enable 恢复正常。"
echo "接下来可以跑完整编排："
echo "  cd /home/fiveages/lerobot_ros2/examples/IsaacSim"
echo "  motion-generation --robot unitree_g1d --task-key pick_tube_full --scene lift"
echo "=================================================================="
