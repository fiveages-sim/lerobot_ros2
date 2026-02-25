#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lerobot-ros2"
DEFAULT_PYTHON_VERSION="3.12"
PYTHON_VERSION="${PYTHON_VERSION:-}"
LEROBOT_SUBMODULE_PATH="submodules/lerobot"
LEROBOT_PINNED_COMMIT="55e752f0c2e7fab0d989c5ff999fbe3b6d8872ab"

print_usage() {
  echo "用法: $0 [submodules|lerobot|conda [python版本]|install|install-plugins|conda-runtime|pypi-mirror|all [python版本]]"
  echo
  echo "不带参数时进入交互菜单。"
  echo "未指定版本时默认使用 Python $DEFAULT_PYTHON_VERSION。"
}

init_submodules() {
  local submodule_paths=()
  local path_line

  echo ">>> 初始化子模块..."
  git -C "$ROOT_DIR" submodule update --init --recursive

  while IFS= read -r path_line; do
    submodule_paths+=("$path_line")
  done < <(git -C "$ROOT_DIR" config --file .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

  echo ">>> 切换常规子模块到最新 main 分支..."
  for submodule_path in "${submodule_paths[@]}"; do
    local submodule_dir="$ROOT_DIR/$submodule_path"

    if [[ "$submodule_path" == "$LEROBOT_SUBMODULE_PATH" ]]; then
      echo ">>> 跳过子模块: $submodule_path（该模块固定版本，使用 lerobot 选项单独初始化）"
      continue
    fi

    echo ">>> 处理子模块: $submodule_path"

    if ! git -C "$submodule_dir" rev-parse --git-dir >/dev/null 2>&1; then
      echo "    跳过：目录不是有效 Git 仓库"
      continue
    fi

    if ! git -C "$submodule_dir" show-ref --verify --quiet refs/remotes/origin/main; then
      echo "    跳过：未找到 origin/main"
      continue
    fi

    git -C "$submodule_dir" fetch origin main
    if git -C "$submodule_dir" show-ref --verify --quiet refs/heads/main; then
      git -C "$submodule_dir" checkout main
    else
      git -C "$submodule_dir" checkout -b main --track origin/main
    fi
    git -C "$submodule_dir" pull --ff-only origin main
  done

  echo ">>> 常规子模块初始化并切换 main 完成。"
}

init_lerobot_submodule() {
  local lerobot_dir="$ROOT_DIR/$LEROBOT_SUBMODULE_PATH"

  echo ">>> 初始化 lerobot 子模块..."
  git -C "$ROOT_DIR" submodule update --init "$LEROBOT_SUBMODULE_PATH"

  if [[ ! -d "$lerobot_dir" ]]; then
    echo "未找到目录: $lerobot_dir"
    exit 1
  fi

  echo ">>> 固定 lerobot 到指定提交: $LEROBOT_PINNED_COMMIT"
  git -C "$lerobot_dir" fetch --all --tags
  git -C "$lerobot_dir" checkout "$LEROBOT_PINNED_COMMIT"
  echo ">>> lerobot 初始化完成。"
}

resolve_python_version() {
  local input_version="${1:-}"
  if [[ -n "$input_version" ]]; then
    echo "$input_version"
  elif [[ -n "$PYTHON_VERSION" ]]; then
    echo "$PYTHON_VERSION"
  else
    echo "$DEFAULT_PYTHON_VERSION"
  fi
}

with_nounset_disabled() {
  local nounset_was_on=0
  local exit_code=0

  if [[ "$-" == *u* ]]; then
    nounset_was_on=1
    set +u
  fi

  "$@" || exit_code=$?

  if [[ $nounset_was_on -eq 1 ]]; then
    set -u
  fi

  return "$exit_code"
}

activate_target_conda_env() {
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
}

ensure_conda_env_active() {
  local current_env="${CONDA_DEFAULT_ENV:-}"
  local current_prefix_basename=""

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    current_prefix_basename="$(basename "${CONDA_PREFIX}")"
  fi

  if [[ "$current_env" == "$ENV_NAME" || "$current_prefix_basename" == "$ENV_NAME" ]]; then
    echo ">>> 检测到当前已在 conda 环境 '$ENV_NAME'，跳过激活。"
    return 0
  fi

  with_nounset_disabled activate_target_conda_env
}

configure_conda_runtime_libs() {
  local env_prefix=""
  local activate_dir=""
  local deactivate_dir=""
  local activate_script=""
  local deactivate_script=""

  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 不存在，请先创建 conda 环境。"
    exit 1
  fi

  ensure_conda_env_active
  env_prefix="${CONDA_PREFIX:-}"
  if [[ -z "$env_prefix" || ! -d "$env_prefix" ]]; then
    echo "无法确定 conda 环境路径，请确认环境 '$ENV_NAME' 可正常激活。"
    exit 1
  fi

  activate_dir="$env_prefix/etc/conda/activate.d"
  deactivate_dir="$env_prefix/etc/conda/deactivate.d"
  activate_script="$activate_dir/lerobot_ros2_runtime_libs.sh"
  deactivate_script="$deactivate_dir/lerobot_ros2_runtime_libs.sh"
  mkdir -p "$activate_dir" "$deactivate_dir"

  cat > "$activate_script" <<'EOF'
#!/usr/bin/env bash
export _LEROBOT_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export _LEROBOT_OLD_LD_PRELOAD="${LD_PRELOAD:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8:$CONDA_PREFIX/lib/libtiff.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
EOF

  cat > "$deactivate_script" <<'EOF'
#!/usr/bin/env bash
export LD_LIBRARY_PATH="${_LEROBOT_OLD_LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${_LEROBOT_OLD_LD_PRELOAD:-}"
unset _LEROBOT_OLD_LD_LIBRARY_PATH
unset _LEROBOT_OLD_LD_PRELOAD
EOF

  chmod +x "$activate_script" "$deactivate_script"
  echo ">>> 已写入 conda 运行时库配置："
  echo "    激活脚本: $activate_script"
  echo "    反激活脚本: $deactivate_script"
}

create_conda_env() {
  local selected_python_version
  selected_python_version="$(resolve_python_version "${1:-}")"

  echo ">>> 创建 conda 环境: $ENV_NAME (Python $selected_python_version)"

  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 已存在，跳过创建。"
    return 0
  fi

  conda create -n "$ENV_NAME" "python=$selected_python_version" -y
  echo ">>> conda 环境创建完成: $ENV_NAME"
}

install_projects() {
  local interface_dir="$ROOT_DIR/submodules/ros2_robot_interface"

  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 不存在，请先创建 conda 环境。"
    exit 1
  fi

  for project_dir in "$interface_dir"; do
    if [[ ! -d "$project_dir" ]]; then
      echo "未找到目录: $project_dir"
      echo "请先执行子模块初始化。"
      exit 1
    fi
  done

  echo ">>> 激活 conda 环境并安装 interface"
  (
    ensure_conda_env_active
    python -m pip install -e "$interface_dir"
  )
  echo ">>> 安装完成。"
}

install_plugins() {
  local robot_plugin_dir="$ROOT_DIR/lerobot_robot_ros2"
  local camera_plugin_dir="$ROOT_DIR/lerobot_camera_ros2"
  local lerobot_dir="$ROOT_DIR/$LEROBOT_SUBMODULE_PATH"

  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "环境 '$ENV_NAME' 不存在，请先创建 conda 环境。"
    exit 1
  fi

  for project_dir in "$robot_plugin_dir" "$camera_plugin_dir"; do
    if [[ ! -d "$project_dir" ]]; then
      echo "未找到目录: $project_dir"
      exit 1
    fi
  done

  # 插件依赖指定版本的 lerobot，先确保子模块已初始化并固定到目标提交。
  init_lerobot_submodule

  if [[ ! -d "$lerobot_dir" ]]; then
    echo "未找到目录: $lerobot_dir"
    exit 1
  fi

  echo ">>> 激活 conda 环境并安装 CUDA/PyTorch/ffmpeg 与指定版本 lerobot + 插件包"
  (
    ensure_conda_env_active

    echo ">>> 安装 CUDA Toolkit 12.8（conda）"
    with_nounset_disabled conda install -y -c nvidia cuda-toolkit=12.8

    echo ">>> 安装 PyTorch 2.7.1/cu128（pip）"
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

    echo ">>> 安装 ffmpeg（conda-forge）"
    with_nounset_disabled conda install -y ffmpeg -c conda-forge

    echo ">>> 预装 evdev（conda-forge，避免 pip 本地编译 evdev）"
    with_nounset_disabled conda install -y evdev -c conda-forge

    echo ">>> 固定 NumPy < 2（避免与 ROS/cv_bridge 的 ABI 冲突）"
    python -m pip install "numpy<2"

    echo ">>> 安装指定版本 lerobot 与插件包"
    python -m pip install -e "$lerobot_dir"
    python -m pip install -e "$robot_plugin_dir" --no-deps
    python -m pip install -e "$camera_plugin_dir"

    echo ">>> 再次确认 NumPy < 2（防止被依赖解析升级到 2.x）"
    python -m pip install "numpy<2"
  )
  configure_conda_runtime_libs
  echo ">>> CUDA/PyTorch/ffmpeg、指定版本 lerobot 与插件安装完成。"
}

configure_nju_pypi_mirror() {
  local pip_config_dir="$HOME/.config/pip"
  local pip_config_file="$pip_config_dir/pip.conf"

  mkdir -p "$pip_config_dir"

  if [[ -f "$pip_config_file" ]]; then
    cp "$pip_config_file" "$pip_config_file.bak.$(date +%Y%m%d%H%M%S)"
    echo ">>> 已备份现有配置: $pip_config_file.bak.<timestamp>"
  fi

  cat > "$pip_config_file" <<'EOF'
[global]
index-url = https://mirrors.nju.edu.cn/pypi/web/simple
format = columns
EOF

  echo ">>> 已配置 PyPI 镜像为 NJU: https://mirrors.nju.edu.cn/pypi/web/simple"
  echo ">>> 配置文件: $pip_config_file"
}

run_all() {
  local python_version="${1:-}"
  init_submodules
  create_conda_env "$python_version"
  install_projects
}

main() {
  local python_version_arg="${2:-}"
  case "${1:-}" in
    submodules)
      init_submodules
      ;;
    lerobot)
      init_lerobot_submodule
      ;;
    conda)
      create_conda_env "$python_version_arg"
      ;;
    install)
      install_projects
      ;;
    install-plugins)
      install_plugins
      ;;
    conda-runtime)
      configure_conda_runtime_libs
      ;;
    pypi-mirror)
      configure_nju_pypi_mirror
      ;;
    all)
      run_all "$python_version_arg"
      ;;
    "")
      echo "请选择操作:"
      echo "  1) 初始化子模块"
      echo "  2) 初始化 lerobot（固定提交）"
      echo "  3) 创建 lerobot-ros2 conda 环境"
      echo "  4) 安装 interface"
      echo "  5) 安装 lerobot 插件（含 OpenCV 兼容参数）"
      echo "  6) 全部执行"
      echo "  7) 配置 NJU PyPI 镜像"
      echo "  8) 配置 conda 运行时库环境（LD_LIBRARY_PATH/LD_PRELOAD）"
      echo "  q) 退出"
      read -r -p "输入选项 [1/2/3/4/5/6/7/8/q]: " choice
      case "$choice" in
        1) init_submodules ;;
        2)
          init_lerobot_submodule
          ;;
        3)
          read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
          create_conda_env "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
          ;;
        4)
          install_projects
          ;;
        5)
          install_plugins
          ;;
        6)
          read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
          run_all "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
          ;;
        7)
          configure_nju_pypi_mirror
          ;;
        8)
          configure_conda_runtime_libs
          ;;
        q|Q) echo "已退出。" ;;
        *) echo "无效选项。"; exit 1 ;;
      esac
      ;;
    -h|--help|help)
      print_usage
      ;;
    *)
      echo "未知参数: $1"
      print_usage
      exit 1
      ;;
  esac
}

main "${1:-}" "${2:-}"
