#!/usr/bin/env bash
# 用 uv 建 lerobot_ros2 的运行时环境（精确复制 conda 的可用组合）。
#
# 为什么不是 `./init.sh all`
# ==========================
# `init.sh all` 会装 PyPI `lerobot==0.5.1` 并强制 `numpy>=2`，而仓库 README
# 自己也写明：NumPy 2 会破坏系统 cv_bridge → **无法采头相机**。
# 本脚本装的是 submodule 里的 lerobot 0.3.4（editable）+ numpy 1.26.4，
# 这个组合实测可同时用于：任务编排、话题采集（含头相机）、LeRobot 导出。
#
# 需要的组合（来自 conda 环境 pip freeze 快照）:
#   numpy 1.26.4  +  cv_bridge OK  +  lerobot 0.3.4(editable)  +  torch 2.7.1+cu128
#
# 用法:
#   cd ~/lerobot_ros2
#   ./scripts/setup_uv_env.sh              # 完整安装
#   ./scripts/setup_uv_env.sh --check      # 只检查现有环境是否满足
#   ./scripts/setup_uv_env.sh --skip-torch # 已有 torch 时跳过（省几分钟）
#
# 注意：**不要**在 Cursor 的 agent sandbox 里跑这个脚本 —— sandbox 会重定向
# UV_CACHE_DIR / TMPDIR，导致包装不进真实 .venv。请在普通终端执行。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${LR_UV_VENV:-.venv}"
PY="$VENV/bin/python"
FROZEN="$ROOT/docs/requirements-uv-frozen.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[uv]${NC} $*"; }
warn() { echo -e "${YELLOW}[uv]${NC} $*"; }
die()  { echo -e "${RED}[uv] ERROR${NC} $*" >&2; exit 1; }

CHECK_ONLY=0
SKIP_TORCH=0
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    --skip-torch) SKIP_TORCH=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) die "未知参数: $a" ;;
  esac
done

command -v uv >/dev/null || die "没有 uv。装: curl -LsSf https://astral.sh/uv/install.sh | sh"

# ---------------------------------------------------------------- 检查模式
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  [[ -x "$PY" ]] || die "$PY 不存在，先跑不带 --check 的版本"
  echo "=== 环境检查 ($VENV) ==="
  # rclpy / cv_bridge / cv2 来自系统 ROS，必须先 source 才能校验
  ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
  ( set +u
    [[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP" >/dev/null 2>&1
    "$PY" - <<'PYEOF'
import sys
mods = [
    ("numpy", "numpy"), ("torch", "torch"), ("lerobot", "lerobot"),
    ("pyarrow", "pyarrow"), ("av", "av"), ("cv_bridge", "cv_bridge"),
    ("rclpy", "rclpy"),
]
bad = []
for label, mod in mods:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "")
        loc = getattr(m, "__file__", "") or ""
        print(f"  ✅ {label:12s} {ver:18s} {loc}")
    except Exception as exc:
        print(f"  ❌ {label:12s} {type(exc).__name__}: {str(exc)[:80]}")
        bad.append(label)
import numpy as np
print()
if np.__version__.startswith("2."):
    print("  ⚠ numpy 是 2.x —— cv_bridge 做数组转换时可能踩 ABI 问题")
try:
    import torch
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
except Exception:
    pass
try:
    import cv2
    print(f"  cv2 {cv2.__version__}  （应来自系统 ROS，不是 pip opencv）")
except Exception:
    print("  cv2 缺失")
print()
print("结论:", "全部就绪 ✅" if not bad else f"缺 {bad} ❌")
sys.exit(0 if not bad else 1)
PYEOF
  )
  exit $?
fi

# ---------------------------------------------------------------- 安装
[[ -f "$FROZEN" ]] || die "找不到依赖清单: $FROZEN"

info "1/5 创建 venv（**不加 --system-site-packages**）"
# rclpy / cv_bridge / cv2 等 ROS 包由 `source /opt/ros/jazzy/setup.bash` 通过
# PYTHONPATH 提供，与 venv 无关。加 --system-site-packages 反而会把系统
# numpy / opencv 混进来，与 venv 内的版本打架（实测系统 cv2 是 4.6.0，
# pip 的 opencv-python-headless 是 4.13 且要求 numpy>=2）。
#
# 只有在 .venv 不存在时才创建 —— 避免把已装好的 torch（2.5GB）清掉。
if [[ ! -x "$PY" ]]; then
  uv venv --python 3.12 "$VENV" || die "建 venv 失败"
else
  info "   $VENV 已存在，保留（要重建请先 rm -rf $VENV）"
fi

if [[ "$SKIP_TORCH" -eq 0 ]]; then
  # 已装就跳过 —— torch 2.5GB，重装很慢（而且 uv 会因 --clear 重下）
  if "$PY" - <<'PYEOF' 2>/dev/null
import sys
try:
    import torch
    sys.exit(0 if torch.__version__.startswith("2.7.1") else 1)
except ImportError:
    sys.exit(1)
PYEOF
  then
    info "2/5 torch 2.7.1 已安装，跳过"
  else
    info "2/5 安装 PyTorch 2.7.1 + cu128（约 2.5GB，首次较慢）"
    uv pip install --python "$PY" \
      "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
      --index-url "$TORCH_INDEX" || die "torch 安装失败（网络？换 UV_INDEX_URL 镜像试试）"

    # **torch 会把自己依赖的 numpy 2.x 一起装进来**（本例装成 2.5.4）。
    # 而我们要 numpy 1.26.4 —— cv_bridge 与 ROS 系统 OpenCV 都按 1.x 编译的。
    # 虽然实测 numpy 2.5.4 下 cv_bridge 能 import（只报警告），但真正做
    # numpy 数组转换时容易踩 ABI 问题，所以这里立刻钉回 1.26.4。
    info "   把 numpy 钉回 1.26.4（torch 会带进 2.x）"
    uv pip install --python "$PY" "numpy==1.26.4" || die "numpy 降级失败"
  fi
else
  info "2/5 跳过 torch（--skip-torch）"
fi

info "3/5 安装其余依赖（精确版本，见 $FROZEN）"
# 清单已剔除三类**不能从 PyPI 装**的包：
#   1) ROS 系统包（ackermann_msgs / control_msgs / cv_bridge … 183 个）
#      —— 由 /opt/ros/jazzy 提供，通过 source setup.bash 可见
#   2) isaac_jazzy_ws 本地工作空间包（isaac-* / cmdvel-to-ackermann … 10 个）
#      —— 由 isaac_jazzy_ws/install 提供
#   3) opencv-python-headless
#      —— 它的 metadata 要求 numpy>=2，与我们要的 1.26.4 冲突；
#         实际 conda 里 cv2 4.13.0 + numpy 1.26.4 是能跑的（conda build 较旧），
#         但 uv 严格按 metadata 检查会拒绝。改用系统 ROS 的 cv2。
#   另剔除 cmake / evdev（后者需本地编译，只游戏手柄遥操用）
uv pip install --python "$PY" -r "$FROZEN" || die "依赖安装失败"

info "4/5 安装本地包（editable，--no-deps 避免去 PyPI 找 rclpy）"
for p in submodules/lerobot submodules/ros2_robot_interface \
         submodules/robot_action_composer lerobot_robot_ros2 lerobot_camera_ros2; do
  [[ -d "$p" ]] || { warn "跳过（目录不存在）: $p"; continue; }
  uv pip install --python "$PY" --no-deps -e "$p" || die "安装 $p 失败"
done

info "5/5 验证"
# 注意：rclpy / cv_bridge / cv2 来自系统 ROS，不装进 venv。
# 校验它们需要先 source ROS —— 这里显式 source 一次，避免误报"缺失"。
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  ( set +u; source "$ROS_SETUP" >/dev/null 2>&1
    "$PY" - <<'PYEOF'
import numpy, torch, pyarrow, av
print(f"  numpy   {numpy.__version__}")
print(f"  torch   {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  pyarrow {pyarrow.__version__}")
print(f"  av      {av.__version__}")
try:
    import cv_bridge; print("  cv_bridge OK（系统 ROS，可采头相机）")
except ImportError:
    print("  cv_bridge MISSING —— source /opt/ros/jazzy/setup.bash 后重试")
try:
    import rclpy; print("  rclpy     OK（系统 ROS）")
except ImportError:
    print("  rclpy     MISSING —— source /opt/ros/jazzy/setup.bash")
try:
    import lerobot; print(f"  lerobot   {lerobot.__file__}")
except ImportError:
    print("  lerobot   MISSING")
if numpy.__version__.startswith("2."):
    print("  ⚠ numpy 2.x —— cv_bridge 的数组转换可能踩 ABI 问题")
PYEOF
  )
else
  warn "找不到 $ROS_SETUP，跳过 rclpy/cv_bridge 校验"
  "$PY" -c "import numpy, torch, pyarrow, av; print('numpy', numpy.__version__); print('torch', torch.__version__)"
fi

echo
info "完成。激活: source $VENV/bin/activate"
info "自检:    ./scripts/setup_uv_env.sh --check"
