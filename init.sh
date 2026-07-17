#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_ENV_BACKEND_OVERRIDE="${LR_ENV_BACKEND:-}"
# shellcheck source=scripts/lr-env.sh
source "${ROOT_DIR}/scripts/lr-env.sh"
LR_ENV_ROOT_DIR="$ROOT_DIR"

DEFAULT_PYTHON_VERSION="3.12"
PYTHON_VERSION="${PYTHON_VERSION:-}"
INSTALL_BACKEND_OVERRIDE=""
ROBOT_ACTION_COMPOSER_SUBMODULE_PATH="submodules/robot_action_composer"

print_usage() {
  echo "用法: $0 [submodules|update-submodules|env [python版本]|install [--conda|--uv]|"
  echo "      install-plugins|install-lerobot|set-backend conda|uv|conda-runtime|"
  echo "      pypi-mirror|ros2-workspace [--all]|all-motion|all [python版本]]"
  echo
  echo "执行链路:"
  echo "  submodules         初始化子模块（git submodule update --init）"
  echo "  update-submodules  将所有子模块更新到 origin/main"
  echo "  env                按 .fa-env.toml 的 backend 创建环境"
  echo "  install            安装任务编排（ros2_robot_interface + robot_action_composer）"
  echo "  install-plugins    安装 PyTorch + PyPI lerobot + 插件包"
  echo "  install-lerobot    同 install-plugins"
  echo "  all-motion         顺序执行 submodules + env + install（仅任务编排）"
  echo "  all                顺序执行 submodules + env + install + install-plugins（全量）"
  echo
  echo "配置:"
  echo "  set-backend        修改 .fa-env.toml 中 backend (conda/uv)"
  echo "  ros2-workspace     写入 .fa-env.toml；按 backend 写 activate 挂钩；--all 则 conda+uv 都写"
  echo "  pypi-mirror        配置 NJU PyPI 镜像"
  echo "  conda-runtime      写入 conda 运行时库挂钩（LD_LIBRARY_PATH/LD_PRELOAD）"
  echo
  echo "配置见 .fa-env.toml；个人覆盖: .fa-env.local.toml；临时: LR_ENV_BACKEND=uv"
  echo "不带参数时进入交互菜单。未指定版本时默认使用 Python $DEFAULT_PYTHON_VERSION。"
}

init_submodules() {
  echo ">>> 初始化子模块..."
  git -C "$ROOT_DIR" submodule update --init --recursive
  echo ">>> 子模块初始化完成。"
}

update_submodules_to_main() {
  local submodule_paths=()
  local path_line

  while IFS= read -r path_line; do
    submodule_paths+=("$path_line")
  done < <(git -C "$ROOT_DIR" config --file .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

  if [[ ${#submodule_paths[@]} -eq 0 ]]; then
    echo ">>> 未找到子模块配置，请先执行子模块初始化。"
    return 0
  fi

  echo ">>> 更新所有子模块到最新 main 分支..."
  for submodule_path in "${submodule_paths[@]}"; do
    local submodule_dir="$ROOT_DIR/$submodule_path"

    echo ">>> 处理子模块: $submodule_path"

    if ! git -C "$submodule_dir" rev-parse --git-dir >/dev/null 2>&1; then
      echo "    跳过：目录不是有效 Git 仓库（可先执行选项 1 初始化）"
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

  echo ">>> 子模块已更新到最新 main。"
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

parse_install_backend_args() {
  INSTALL_BACKEND_OVERRIDE=""
  local arg
  for arg in "$@"; do
    [[ -z "$arg" ]] && continue
    case "$arg" in
      --conda) INSTALL_BACKEND_OVERRIDE="conda" ;;
      --uv) INSTALL_BACKEND_OVERRIDE="uv" ;;
      *)
        echo "未知参数: $arg"
        print_usage
        exit 1
        ;;
    esac
  done
}

create_conda_env() {
  local selected_python_version
  local env_name
  selected_python_version="$(resolve_python_version "${1:-}")"
  lr_env_load_config "$ROOT_DIR"
  env_name="$LR_ENV_CONDA_NAME"

  echo ">>> 创建 conda 环境: $env_name (Python $selected_python_version)"

  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    echo "环境 '$env_name' 已存在，跳过创建。"
    return 0
  fi

  conda create -n "$env_name" "python=$selected_python_version" -y
  echo ">>> conda 环境创建完成: $env_name"
}

create_uv_env() {
  local selected_python_version
  local venv_path
  selected_python_version="$(resolve_python_version "${1:-}")"
  lr_env_load_config "$ROOT_DIR"
  venv_path="$(lr_env_uv_venv_path)"

  echo ">>> 创建 uv 虚拟环境: $venv_path (Python $selected_python_version)"

  if ! command -v uv >/dev/null 2>&1; then
    echo "未检测到 uv，请先安装: https://docs.astral.sh/uv/"
    exit 1
  fi

  if [[ -f "${venv_path}/bin/activate" ]]; then
    echo "虚拟环境已存在: $venv_path，跳过创建。"
    return 0
  fi

  lr_env_create_uv_venv "$selected_python_version" "$venv_path"
  echo ">>> uv 虚拟环境创建完成: $venv_path（--system-site-packages）"
  if [[ -n "${LR_ENV_ROS2_WORKSPACE:-}" ]]; then
    lr_env_write_uv_ros2_hook "$venv_path" "$LR_ENV_ROS2_WORKSPACE"
    echo ">>> 已根据 .fa-env.toml 为 venv 写入 ROS2 activate 挂钩。"
  else
    echo ">>> 提示: 执行 ./init.sh ros2-workspace 后，source .venv/bin/activate 会自动 source ROS2，并注册 ros2-stack 补全。"
  fi
}

create_env_for_configured_backend() {
  local python_version="${1:-}"
  lr_env_load_config "$ROOT_DIR"
  case "$LR_ENV_BACKEND" in
    conda) create_conda_env "$python_version" ;;
    uv) create_uv_env "$python_version" ;;
    *)
      echo "未知 backend: $LR_ENV_BACKEND"
      exit 1
      ;;
  esac
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

  lr_env_load_config "$ROOT_DIR"
  if ! conda env list | awk '{print $1}' | grep -Fxq "$LR_ENV_CONDA_NAME"; then
    echo "环境 '$LR_ENV_CONDA_NAME' 不存在，请先创建 conda 环境。"
    exit 1
  fi

  set +u
  eval "$(conda shell.bash hook)"
  conda activate "$LR_ENV_CONDA_NAME"
  env_prefix="${CONDA_PREFIX:-}"
  if [[ -z "$env_prefix" || ! -d "$env_prefix" ]]; then
    echo "无法确定 conda 环境路径，请确认环境 '$LR_ENV_CONDA_NAME' 可正常激活。"
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

install_projects() {
  local interface_dir="$ROOT_DIR/submodules/ros2_robot_interface"
  local rac_dir="$ROOT_DIR/$ROBOT_ACTION_COMPOSER_SUBMODULE_PATH"

  lr_env_load_config "$ROOT_DIR"
  if [[ -n "$INSTALL_BACKEND_OVERRIDE" ]]; then
    LR_ENV_BACKEND="$INSTALL_BACKEND_OVERRIDE"
  fi

  for project_dir in "$interface_dir" "$rac_dir"; do
    if [[ ! -d "$project_dir" ]]; then
      echo "未找到目录: $project_dir"
      echo "请先执行子模块初始化。"
      exit 1
    fi
  done

  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装 ros2_robot_interface 与 robot_action_composer"
  (
    set +u
    lr_env_activate "$ROOT_DIR"
    lr_env_install_editable "$interface_dir"
    lr_env_install_editable "$rac_dir"
  )
  echo ">>> 安装完成。"
}

install_plugins() {
  local robot_plugin_dir="$ROOT_DIR/lerobot_robot_ros2"
  local camera_plugin_dir="$ROOT_DIR/lerobot_camera_ros2"
  local lerobot_pkg=""

  lr_env_load_config "$ROOT_DIR"
  lerobot_pkg="lerobot==${LR_ENV_LEROBOT_VERSION}"
  if [[ -n "$INSTALL_BACKEND_OVERRIDE" ]]; then
    LR_ENV_BACKEND="$INSTALL_BACKEND_OVERRIDE"
  fi

  for project_dir in "$robot_plugin_dir" "$camera_plugin_dir"; do
    if [[ ! -d "$project_dir" ]]; then
      echo "未找到目录: $project_dir"
      exit 1
    fi
  done

  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装 PyTorch + $lerobot_pkg + 插件包"
  (
    set +u
    lr_env_activate "$ROOT_DIR"

    if [[ "$LR_ENV_BACKEND" == "conda" ]]; then
      echo ">>> 安装 CUDA Toolkit 12.8（conda）"
      with_nounset_disabled conda install -y -c nvidia cuda-toolkit=12.8

      echo ">>> 安装 ffmpeg（conda-forge）"
      with_nounset_disabled conda install -y ffmpeg -c conda-forge

      echo ">>> 预装 evdev（conda-forge，避免 pip 本地编译 evdev）"
      with_nounset_disabled conda install -y evdev -c conda-forge
    else
      echo ">>> [uv] 请确保系统已安装 ffmpeg（如 apt install ffmpeg）"
    fi

    echo ">>> 安装 PyTorch 2.7.1/cu128"
    lr_env_pip_install_torch

    echo ">>> 安装 PyPI lerobot: $lerobot_pkg"
    lr_env_pip_install "$lerobot_pkg"

    echo ">>> 固定 numpy 版本（lerobot 0.5.x 要求 >=2.0,<2.3）"
    lr_env_pip_install "numpy>=2.0,<2.3"

    echo ">>> 安装 scipy（覆盖 system-site-packages 中与 numpy 2.x 不兼容的版本）"
    lr_env_pip_install "scipy>=1.14"

    echo ">>> 安装插件包"
    lr_env_install_editable "$robot_plugin_dir"
    lr_env_install_editable "$camera_plugin_dir"
  )

  if [[ "$LR_ENV_BACKEND" == "conda" ]]; then
    configure_conda_runtime_libs
  fi
  echo ">>> PyTorch、$lerobot_pkg 与插件安装完成。"
}

install_motion_stack() {
  lr_env_load_config "$ROOT_DIR"
  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装任务编排栈（interface + robot_action_composer）"
  install_projects
}

install_lerobot_stack() {
  lr_env_load_config "$ROOT_DIR"
  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装 lerobot 相关（录制 / 推理 / 插件）"
  install_plugins
}

configure_ros2_workspace_source() {
  local ws_input ws_stored apply_all=0 hook_choice
  local arg
  lr_env_load_config "$ROOT_DIR"

  for arg in "$@"; do
    case "$arg" in
      --all) apply_all=1 ;;
      *)
        echo "未知参数: $arg"
        print_usage
        exit 1
        ;;
    esac
  done

  read -r -p "输入 ROS2 工作空间路径（默认 ~/ros2_ws）: " ws_input
  ws_input="${ws_input:-~/ros2_ws}"
  ws_stored="$ws_input"
  lr_env_set_ros2_workspace "$ws_stored"
  echo ">>> 已写入 .fa-env.toml [ros2].workspace = $ws_stored"

  if [[ "$apply_all" -eq 0 ]]; then
    read -r -p "是否为 conda 与 uv 都写入 activate 挂钩？[y/N]: " hook_choice
  fi
  if [[ "$apply_all" -eq 1 || "$hook_choice" =~ ^[Yy]$ ]]; then
    echo ">>> 为 conda 与 uv 环境写入 activate 挂钩..."
    lr_env_apply_ros2_hooks "$ws_stored" 1
  else
    echo ">>> 按 backend=$LR_ENV_BACKEND 写入 activate 挂钩..."
    lr_env_apply_ros2_hooks "$ws_stored" 0
  fi
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

run_all_motion() {
  local python_version="${1:-}"
  init_submodules
  create_env_for_configured_backend "$python_version"
  install_motion_stack
}

run_all_full() {
  local python_version="${1:-}"
  run_all_motion "$python_version"
  install_lerobot_stack
}

main() {
  case "${1:-}" in
    submodules)
      init_submodules
      ;;
    update-submodules)
      update_submodules_to_main
      ;;
    env|conda|uv)
      create_env_for_configured_backend "${2:-}"
      ;;
    set-backend)
      if [[ -z "${2:-}" ]]; then
        echo "用法: $0 set-backend conda|uv"
        exit 1
      fi
      lr_env_set_backend "$2"
      ;;
    install)
      parse_install_backend_args "${@:2}"
      install_motion_stack
      ;;
    install-plugins|install-lerobot)
      parse_install_backend_args "${@:2}"
      install_lerobot_stack
      ;;
    conda-runtime)
      configure_conda_runtime_libs
      ;;
    ros2-workspace)
      configure_ros2_workspace_source "${@:2}"
      ;;
    pypi-mirror)
      configure_nju_pypi_mirror
      ;;
    all-motion)
      run_all_motion "${2:-}"
      ;;
    all)
      run_all_full "${2:-}"
      ;;
    "")
      lr_env_load_config "$ROOT_DIR"
      echo "请选择操作:"
      echo "  当前 backend: $LR_ENV_BACKEND（见 .fa-env.toml）"
      echo "  1) 初始化子模块"
      echo "  2) 按当前 backend 创建环境"
      echo "  3) 安装任务编排（ros2_robot_interface + robot_action_composer）"
      echo "  4) 安装 lerobot 相关（PyTorch + lerobot + 插件包）"
      echo "  -------------------- 执行链路 --------------------"
      echo "  5) 全部执行（任务编排：1 + 2 + 3）"
      echo "  6) 全部执行（任务编排 + lerobot：1 + 2 + 3 + 4）"
      echo "  7) 更新所有子模块到最新 main"
      echo "  -------------------- 配置 --------------------"
      echo "  8) 配置 NJU PyPI 镜像"
      echo "  9) 配置 ROS2 工作空间（.fa-env.toml + 可选 conda hook）"
      echo "  10) 切换 backend (conda/uv)"
      echo "  q) 退出"
      read -r -p "输入选项 [1-10/q]: " choice
      case "$choice" in
        1) init_submodules ;;
        2)
          read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
          create_env_for_configured_backend "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
          ;;
        3) install_motion_stack ;;
        4) install_lerobot_stack ;;
        5)
          read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
          run_all_motion "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
          ;;
        6)
          read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
          run_all_full "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
          ;;
        7) update_submodules_to_main ;;
        8) configure_nju_pypi_mirror ;;
        9) configure_ros2_workspace_source ;;
        10)
          read -r -p "输入 backend [conda/uv]（当前 $LR_ENV_BACKEND）: " backend_choice
          backend_choice="${backend_choice:-$LR_ENV_BACKEND}"
          lr_env_set_backend "$backend_choice"
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

main "$@"
