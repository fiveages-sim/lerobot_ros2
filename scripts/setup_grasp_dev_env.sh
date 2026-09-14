#!/usr/bin/env bash
# Setup / refresh the grasp + ros2-stack env in THIS repo only.
# Usage (from repo root):
#   bash scripts/setup_grasp_dev_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo ">>> Creating .venv (Python 3.12, --system-site-packages for ROS rclpy)"
  uv venv --python python3.12 --system-site-packages .venv
fi

echo ">>> Installing ros2_robot_interface + robot_action_composer (--no-deps) + viser"
uv pip install -e submodules/ros2_robot_interface --no-deps --python .venv/bin/python
# System OpenCV (cv2) on Ubuntu/Jazzy is built against NumPy 1.x — do NOT install numpy 2.x
uv pip install "numpy>=1.26,<2" pyyaml --python .venv/bin/python
uv pip install -e submodules/robot_action_composer --no-deps --python .venv/bin/python
uv pip install "viser>=0.2" --python .venv/bin/python

HOOK="${ROOT}/.venv/bin/lr_ros2_workspace.sh"
cat > "${HOOK}" <<'EOF'
#!/usr/bin/env bash
# Sourced by .venv/bin/activate (this repo only).
set +u
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
if [[ -f "${HOME}/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/ros2_ws/install/setup.bash"
fi
_LR_ROS2_WS="${HOME}/fa_w2_ws"
if [[ -f "${_LR_ROS2_WS}/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${_LR_ROS2_WS}/install/setup.bash"
  echo "[venv activate] Sourced ROS2: jazzy + ros2_ws + ${_LR_ROS2_WS}"
elif [[ -n "${ROS_DISTRO:-}" ]]; then
  echo "[venv activate] Sourced ROS2 distro=${ROS_DISTRO}"
else
  echo "[venv activate] WARN: ROS2 setup not found"
fi
if [[ -n "${BASH_VERSION:-}" ]] && command -v ros2-stack >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  eval "$(ros2-stack completion bash 2>/dev/null)" && \
    echo "[venv activate] Registered ros2-stack bash completion" || true
fi
EOF
chmod +x "${HOOK}"

ACT="${ROOT}/.venv/bin/activate"
if ! grep -q 'lr_ros2_workspace.sh' "${ACT}"; then
  cat >> "${ACT}" <<'EOF'

# ROS2 + ros2-stack completion (this checkout only)
if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/lr_ros2_workspace.sh" ]]; then
    # shellcheck disable=SC1091
    . "${VIRTUAL_ENV}/bin/lr_ros2_workspace.sh"
fi
EOF
  echo ">>> Patched .venv/bin/activate"
fi

mkdir -p "${ROOT}"
if [[ ! -f "${ROOT}/.fa-env.local.toml" ]]; then
  cat > "${ROOT}/.fa-env.local.toml" <<'EOF'
# Local overrides for THIS checkout only (gitignored).
[ros2]
workspace = "~/fa_w2_ws"
EOF
fi

echo
echo "OK. Develop only in: ${ROOT}"
echo "  source ${ROOT}/.venv/bin/activate"
echo "  which ros2-stack grasp-generation"
echo "  cd examples/IsaacSim && ros2-stack launch --robot fiveages_w2"
echo "  grasp-generation serve --robot fiveages_w2 --workspace ."
echo
echo "HUG stays separate: submodules/hug/.venv (Python 3.10)."
