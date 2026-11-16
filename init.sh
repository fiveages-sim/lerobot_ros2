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

# 任务编排 / 灵巧手预设路径（勾选默认与快捷键用）
SUBMODULE_CORE_PATHS=(
  "submodules/ros2_robot_interface"
  "submodules/robot_action_composer"
)
SUBMODULE_DEX_PATHS=(
  "submodules/hug"
  "submodules/wuji-retargeting"
)

CATALOG_PATH=()
CATALOG_SEL=()
CATALOG_N=0

print_usage() {
  echo "用法: $0 [submodules [--all|路径...]|submodules-core|submodules-dex|update-submodules|"
  echo "      env [python版本]|install [--conda|--uv]|install-checked|install-grasp|"
  echo "      install-plugins|install-lerobot|hug-env|doctor|"
  echo "      set-backend conda|uv|install-uv|install-miniconda|"
  echo "      pypi-mirror|uv-mirror|conda-runtime|ros2-workspace [--all]|"
  echo "      all-motion|all-grasp|all [python版本]]"
  echo
  echo "执行链路（灵巧手可选，不装也能用任务编排）:"
  echo "  submodules         交互勾选要初始化的子模块（默认勾选 core）"
  echo "  submodules --all   不交互，初始化全部顶层子模块"
  echo "  submodules <路径>  不交互，只初始化指定路径（可多个）"
  echo "  submodules-core    只初始化 ros2_robot_interface 与 robot_action_composer"
  echo "  submodules-dex     初始化 hug 与 wuji-retargeting（不递归嵌套 MuJoCo）"
  echo "  update-submodules  将已检出的子模块更新到 origin/main（跳过未初始化）"
  echo "  env                按 .fa-env.toml 的 backend 创建环境"
  echo "  install            安装任务编排（interface + composer）。不含 viser / wuji-sdk / HUG"
  echo "  install-checked    按已检出子模块安装对应依赖（core / roboapi-fasim / dex）"
  echo "  install-grasp      可选：父仓环境安装 viser + wuji-sdk"
  echo "  install-plugins    安装 PyTorch + PyPI lerobot + 插件包"
  echo "  install-lerobot    同 install-plugins"
  echo "  hug-env            可选：HUG 独立 venv（Python 3.10）。权重可下，MANO 需自行注册"
  echo "  doctor             检查各模块是否就绪，并给出补齐命令（不安装）"
  echo "  all-motion         编排：core 子模块 + 环境 + install（不含灵巧手）"
  echo "  all-grasp          all-motion + dex 子模块 + install-grasp + hug-env"
  echo "  all                all-motion + install-plugins（录制/推理；仍不含 HUG）"
  echo
  echo "配置:"
  echo "  set-backend        修改 .fa-env.toml 中 backend (conda/uv)"
  echo "  install-uv         安装 uv 包管理器（https://astral.sh/uv/）"
  echo "  install-miniconda  安装 Miniconda 到 ~/miniconda3 并关闭 base 自动激活"
  echo "  pypi-mirror        配置 pip 使用 NJU PyPI 镜像"
  echo "  uv-mirror          配置 uv 使用清华 PyPI 镜像（backend=uv 时生效）"
  echo "  ros2-workspace     写入 .fa-env.toml；按 backend 写 activate 挂钩；--all 则 conda+uv 都写"
  echo "  conda-runtime      写入 conda 运行时库挂钩（LD_LIBRARY_PATH/LD_PRELOAD）"
  echo
  echo "配置见 .fa-env.toml；个人覆盖: .fa-env.local.toml；临时: LR_ENV_BACKEND=uv"
  echo "不带参数时进入交互菜单（↑/↓ 选择，Enter 确认）。未指定版本时默认使用 Python $DEFAULT_PYTHON_VERSION。"
}

is_core_submodule_path() {
  local path="$1" p
  for p in "${SUBMODULE_CORE_PATHS[@]}"; do
    [[ "$path" == "$p" ]] && return 0
  done
  return 1
}

is_dex_submodule_path() {
  local path="$1" p
  for p in "${SUBMODULE_DEX_PATHS[@]}"; do
    [[ "$path" == "$p" ]] && return 0
  done
  return 1
}

is_git_worktree() {
  local dir="$1"
  local abs toplevel
  [[ -d "$dir" ]] || return 1
  abs=$(cd "$dir" && pwd -P) || return 1
  toplevel=$(git -C "$abs" rev-parse --show-toplevel 2>/dev/null) || return 1
  [[ "$abs" == "$toplevel" ]]
}

list_gitmodules_paths() {
  git -C "$ROOT_DIR" config --file .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null \
    | awk '{print $2}'
}

build_submodule_catalog() {
  CATALOG_PATH=()
  CATALOG_SEL=()
  CATALOG_N=0
  local path_line
  while IFS= read -r path_line; do
    [[ -z "$path_line" ]] && continue
    CATALOG_PATH+=("$path_line")
    if is_core_submodule_path "$path_line"; then
      CATALOG_SEL+=(1)
    else
      CATALOG_SEL+=(0)
    fi
    CATALOG_N=$((CATALOG_N + 1))
  done < <(list_gitmodules_paths)

  if [[ "$CATALOG_N" -eq 0 ]]; then
    echo ">>> 未找到子模块配置（.gitmodules）。"
    return 1
  fi
  return 0
}

catalog_select_none() {
  local i
  for ((i = 0; i < CATALOG_N; i++)); do
    CATALOG_SEL[$i]=0
  done
}

catalog_select_all() {
  local i
  for ((i = 0; i < CATALOG_N; i++)); do
    CATALOG_SEL[$i]=1
  done
}

catalog_select_core() {
  local i
  for ((i = 0; i < CATALOG_N; i++)); do
    if is_core_submodule_path "${CATALOG_PATH[$i]}"; then
      CATALOG_SEL[$i]=1
    else
      CATALOG_SEL[$i]=0
    fi
  done
}

catalog_select_core_dex() {
  local i
  for ((i = 0; i < CATALOG_N; i++)); do
    if is_core_submodule_path "${CATALOG_PATH[$i]}" || is_dex_submodule_path "${CATALOG_PATH[$i]}"; then
      CATALOG_SEL[$i]=1
    else
      CATALOG_SEL[$i]=0
    fi
  done
}

draw_submodule_catalog() {
  local cursor_idx="$1"
  local i mark cursor
  for ((i = 0; i < CATALOG_N; i++)); do
    if [[ "${CATALOG_SEL[$i]}" == "1" ]]; then
      mark="[x]"
    else
      mark="[ ]"
    fi
    if [[ "$i" -eq "$cursor_idx" ]]; then
      cursor=">"
      printf " %s %s  %s\n" "$cursor" "$mark" "${CATALOG_PATH[$i]}"
    else
      printf "   %s  %s\n" "$mark" "${CATALOG_PATH[$i]}"
    fi
  done
}

pick_submodules_tty() {
  local idx=0 key k2 k3 total
  local old_stty
  old_stty=$(stty -g)
  restore_picker() {
    stty "$old_stty" 2>/dev/null || true
    printf '\033[?25h'
  }
  trap restore_picker EXIT INT TERM
  stty -echo -icanon
  printf '\033[?25l'
  echo "请勾选要初始化的子模块（默认: core）"
  echo "  ↑/↓ 移动  空格 勾选  a 全选  c 仅 core  d core+dex  Enter 确认  q 取消"
  echo ""
  total=$((3 + CATALOG_N))
  draw_submodule_catalog "$idx"
  while true; do
    IFS= read -rsn1 key || key=""
    if [[ "$key" == $'\x1b' ]]; then
      IFS= read -rsn1 -t 0.05 k2 || k2=""
      IFS= read -rsn1 -t 0.05 k3 || k3=""
      if [[ "$k3" == "A" || "$k2" == "A" ]]; then
        idx=$(( (idx - 1 + CATALOG_N) % CATALOG_N ))
      elif [[ "$k3" == "B" || "$k2" == "B" ]]; then
        idx=$(( (idx + 1) % CATALOG_N ))
      fi
    elif [[ "$key" == " " ]]; then
      CATALOG_SEL[$idx]=$((1 - CATALOG_SEL[$idx]))
    elif [[ "$key" == "a" || "$key" == "A" ]]; then
      catalog_select_all
    elif [[ "$key" == "c" || "$key" == "C" ]]; then
      catalog_select_core
    elif [[ "$key" == "d" || "$key" == "D" ]]; then
      catalog_select_core_dex
    elif [[ "$key" == "q" || "$key" == "Q" ]]; then
      restore_picker
      trap - EXIT INT TERM
      echo ""
      echo "已取消。"
      return 1
    elif [[ -z "$key" ]]; then
      restore_picker
      trap - EXIT INT TERM
      echo ""
      break
    fi
    printf '\033[%sA' "$total"
    echo "请勾选要初始化的子模块（默认: core）"
    echo "  ↑/↓ 移动  空格 勾选  a 全选  c 仅 core  d core+dex  Enter 确认  q 取消"
    echo ""
    draw_submodule_catalog "$idx"
  done
  return 0
}

pick_submodules_notty() {
  local i mark input n
  echo "请勾选要初始化的子模块（非交互终端）"
  echo "  回车=仅 core    a=全选    序号=在 core 之外额外勾选"
  echo ""
  for ((i = 0; i < CATALOG_N; i++)); do
    if [[ "${CATALOG_SEL[$i]}" == "1" ]]; then
      mark="[x]"
    else
      mark="[ ]"
    fi
    printf "  %2d) %s  %s\n" "$((i + 1))" "$mark" "${CATALOG_PATH[$i]}"
  done
  echo ""
  read -r -p "输入序号（空格分隔）: " input
  case "$input" in
    a|A) catalog_select_all ;;
    "") catalog_select_core ;;
    *)
      catalog_select_core
      for n in $input; do
        if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -ge 1 ]] && [[ "$n" -le "$CATALOG_N" ]]; then
          CATALOG_SEL[$((n - 1))]=1
        else
          echo "    忽略无效序号: $n"
        fi
      done
      ;;
  esac
  return 0
}

pick_submodules() {
  if ! build_submodule_catalog; then
    return 1
  fi
  if [[ -t 0 && -t 1 ]]; then
    pick_submodules_tty || return 1
  else
    pick_submodules_notty || return 1
  fi

  local i selected=0
  echo ">>> 将初始化以下子模块："
  for ((i = 0; i < CATALOG_N; i++)); do
    if [[ "${CATALOG_SEL[$i]}" == "1" ]]; then
      echo "    - ${CATALOG_PATH[$i]}"
      selected=$((selected + 1))
    fi
  done
  if [[ "$selected" -eq 0 ]]; then
    echo ">>> 未勾选任何子模块，跳过。"
    return 0
  fi
  return 0
}

init_selected_submodules() {
  local paths=("$@")
  if [[ ${#paths[@]} -eq 0 ]]; then
    echo ">>> 未指定子模块路径，跳过。"
    return 0
  fi
  echo ">>> 同步并初始化选中的子模块（追加，不卸载已有）..."
  git -C "$ROOT_DIR" submodule sync -- "${paths[@]}"
  git -C "$ROOT_DIR" submodule update --init -- "${paths[@]}"
  echo ">>> 子模块初始化完成。"
}

init_submodules_from_catalog_selection() {
  local i
  local paths=()
  for ((i = 0; i < CATALOG_N; i++)); do
    if [[ "${CATALOG_SEL[$i]}" == "1" ]]; then
      paths+=("${CATALOG_PATH[$i]}")
    fi
  done
  init_selected_submodules "${paths[@]}"
}

init_submodules_interactive() {
  local install_choice
  pick_submodules || return 1
  init_submodules_from_catalog_selection
  if [[ -t 0 && -t 1 ]]; then
    read -r -p "是否按已检出子模块安装对应依赖？[y/N]: " install_choice
    case "$install_choice" in
      y|Y|yes|YES)
        install_checked_submodules
        ;;
    esac
  fi
}

init_submodules_all() {
  if ! build_submodule_catalog; then
    return 1
  fi
  catalog_select_all
  init_submodules_from_catalog_selection
}

init_submodules_by_paths() {
  local requested=("$@")
  local path known match
  local resolved=()

  if ! build_submodule_catalog; then
    return 1
  fi

  for path in "${requested[@]}"; do
    match=""
    for known in "${CATALOG_PATH[@]}"; do
      if [[ "$path" == "$known" || "$path" == "${known#submodules/}" || "submodules/$path" == "$known" ]]; then
        match="$known"
        break
      fi
    done
    if [[ -z "$match" ]]; then
      echo ">>> 未知子模块路径: $path"
      echo "    可选: ${CATALOG_PATH[*]}"
      return 1
    fi
    resolved+=("$match")
  done

  init_selected_submodules "${resolved[@]}"
}

run_submodules_command() {
  local args=("$@")
  if [[ ${#args[@]} -eq 0 ]]; then
    init_submodules_interactive
    return $?
  fi
  if [[ ${#args[@]} -eq 1 && "${args[0]}" == "--all" ]]; then
    init_submodules_all
    return $?
  fi
  init_submodules_by_paths "${args[@]}"
}

init_submodules_core() {
  echo ">>> 初始化任务编排子模块（ros2_robot_interface、robot_action_composer）..."
  git -C "$ROOT_DIR" submodule update --init -- \
    "${SUBMODULE_CORE_PATHS[@]}"
  echo ">>> 任务编排子模块完成。"
}

init_submodules_dex() {
  echo ">>> 初始化灵巧手子模块（hug、wuji-retargeting；不递归嵌套仓库）..."
  git -C "$ROOT_DIR" submodule update --init -- \
    "${SUBMODULE_DEX_PATHS[@]}"
  echo ">>> 未拉取 wuji-retargeting 里的 mujoco-sim / wuji-description。"
  echo "    产品路径是父仓里的 wuji-sdk（./init.sh install-grasp），不是嵌套仿真。"
  echo "    只在调开源算法时: bash scripts/setup_wuji_retargeting.sh --recursive-nested"
  echo ">>> git 检出不含 HUG 权重和 MANO。权重: ./init.sh hug-env ；MANO 需注册下载。"
}

# 兼容旧名：一键流程不再调用；保留以免外部脚本引用
init_submodules() {
  init_submodules_interactive
}

update_submodules_to_main() {
  local submodule_paths=()
  local path_line
  local updated=0
  local skipped=0

  while IFS= read -r path_line; do
    [[ -n "$path_line" ]] && submodule_paths+=("$path_line")
  done < <(list_gitmodules_paths)

  if [[ ${#submodule_paths[@]} -eq 0 ]]; then
    echo ">>> 未找到子模块配置，请先执行子模块初始化。"
    return 0
  fi

  echo ">>> 更新已检出的子模块到最新 main（跳过未初始化）..."
  for submodule_path in "${submodule_paths[@]}"; do
    local submodule_dir="$ROOT_DIR/$submodule_path"

    if ! is_git_worktree "$submodule_dir"; then
      echo ">>> 跳过未检出: $submodule_path"
      skipped=$((skipped + 1))
      continue
    fi

    echo ">>> 处理子模块: $submodule_path"

    if ! git -C "$submodule_dir" show-ref --verify --quiet refs/remotes/origin/main; then
      git -C "$submodule_dir" fetch origin main 2>/dev/null || true
    fi

    if ! git -C "$submodule_dir" show-ref --verify --quiet refs/remotes/origin/main; then
      echo "    跳过：未找到 origin/main"
      skipped=$((skipped + 1))
      continue
    fi

    git -C "$submodule_dir" fetch origin main
    if git -C "$submodule_dir" show-ref --verify --quiet refs/heads/main; then
      git -C "$submodule_dir" checkout main
    else
      git -C "$submodule_dir" checkout -b main --track origin/main
    fi
    git -C "$submodule_dir" pull --ff-only origin main
    updated=$((updated + 1))
  done

  echo ">>> 子模块更新完成（已更新 $updated，跳过 $skipped）。"
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

_activate_install_env() {
  lr_env_load_config "$ROOT_DIR"
  if [[ -n "$INSTALL_BACKEND_OVERRIDE" ]]; then
    LR_ENV_BACKEND="$INSTALL_BACKEND_OVERRIDE"
  fi
}

install_motion_projects() {
  local interface_dir="$ROOT_DIR/submodules/ros2_robot_interface"
  local rac_dir="$ROOT_DIR/$ROBOT_ACTION_COMPOSER_SUBMODULE_PATH"

  _activate_install_env

  for project_dir in "$interface_dir" "$rac_dir"; do
    if [[ ! -d "$project_dir" ]]; then
      echo "未找到目录: $project_dir"
      echo "请先执行: ./init.sh submodules-core"
      exit 1
    fi
  done

  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装 ros2_robot_interface 与 robot_action_composer"
  echo "    这一步不含 viser / wuji-sdk / HUG。灵巧手另走 install-grasp 与 hug-env。"
  (
    set +u
    lr_env_activate "$ROOT_DIR"
    lr_env_install_editable "$interface_dir"
    lr_env_install_editable "$rac_dir"
    # ROS Jazzy cv_bridge is built for NumPy 1.x. An unpinned numpy>=1.20
    # resolves to 2.x and crashes head-camera capture.
    echo ">>> 固定 numpy<2（与系统 cv_bridge 一致）。不要在这个环境里跑 install-lerobot。"
    lr_env_pip_install "numpy>=1.26,<2"
  )
  echo ">>> 任务编排安装完成（motion-generation / ros2-stack / grasp-generation 入口）。"
  echo "    grasp-generation 命令已在，但 HUG 推理与手型 retarget 未装时会提示缺什么。"
}

install_grasp_extras() {
  _activate_install_env
  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装可选抓取 extras: viser + wuji-sdk"
  echo "    装进父仓环境。不要装进 submodules/hug/.venv。"
  (
    set +u
    lr_env_activate "$ROOT_DIR"
    lr_env_pip_install "viser>=0.2"
    lr_env_pip_install "wuji-sdk>=0.10.0"
    echo ">>> 再次固定 numpy<2（依赖可能把它升到 2.x，头相机采图会挂）。"
    lr_env_pip_install "numpy>=1.26,<2"
  )
  echo ">>> 抓取 extras 完成。HUG 权重与 MANO 仍要 ./init.sh hug-env（MANO 不能自动下）。"
}

install_roboapi_fasim() {
  local fasim_dir="$ROOT_DIR/submodules/roboapi-fasim"
  _activate_install_env
  if ! is_git_worktree "$fasim_dir"; then
    echo "未检出 submodules/roboapi-fasim。请先 ./init.sh submodules"
    return 1
  fi
  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装 roboapi-fasim（editable）"
  (
    set +u
    lr_env_activate "$ROOT_DIR"
    lr_env_install_editable "$fasim_dir"
    echo ">>> 固定 numpy<2（与系统 cv_bridge / 编排栈一致）。"
    lr_env_pip_install "numpy>=1.26,<2"
  )
  echo ">>> roboapi-fasim 安装完成。"
}

install_checked_submodules() {
  local path_line
  local have_interface=0 have_composer=0 have_hug=0 have_wuji=0 have_fasim=0
  local skipped=()
  local actions=0

  echo ">>> 按已检出子模块安装对应依赖..."
  while IFS= read -r path_line; do
    [[ -z "$path_line" ]] && continue
    if ! is_git_worktree "$ROOT_DIR/$path_line"; then
      continue
    fi
    case "$path_line" in
      submodules/ros2_robot_interface) have_interface=1 ;;
      submodules/robot_action_composer) have_composer=1 ;;
      submodules/hug) have_hug=1 ;;
      submodules/wuji-retargeting) have_wuji=1 ;;
      submodules/roboapi-fasim) have_fasim=1 ;;
      *)
        skipped+=("$path_line")
        ;;
    esac
  done < <(list_gitmodules_paths)

  if [[ "$have_interface" -eq 1 && "$have_composer" -eq 1 ]]; then
    install_motion_stack
    actions=$((actions + 1))
  elif [[ "$have_interface" -eq 1 || "$have_composer" -eq 1 ]]; then
    _activate_install_env
    echo ">>> core 未齐（需要 interface + composer），仅安装已检出的编排包..."
    (
      set +u
      lr_env_activate "$ROOT_DIR"
      if [[ "$have_interface" -eq 1 ]]; then
        lr_env_install_editable "$ROOT_DIR/submodules/ros2_robot_interface"
      else
        echo "    跳过未检出: submodules/ros2_robot_interface"
      fi
      if [[ "$have_composer" -eq 1 ]]; then
        lr_env_install_editable "$ROOT_DIR/$ROBOT_ACTION_COMPOSER_SUBMODULE_PATH"
      else
        echo "    跳过未检出: $ROBOT_ACTION_COMPOSER_SUBMODULE_PATH"
      fi
      lr_env_pip_install "numpy>=1.26,<2"
    )
    actions=$((actions + 1))
  fi

  if [[ "$have_fasim" -eq 1 ]]; then
    install_roboapi_fasim
    actions=$((actions + 1))
  fi

  if [[ "$have_hug" -eq 1 || "$have_wuji" -eq 1 ]]; then
    install_grasp_extras
    actions=$((actions + 1))
  fi

  if [[ "$have_hug" -eq 1 ]]; then
    setup_hug_env
    actions=$((actions + 1))
  elif [[ "$have_wuji" -eq 1 ]]; then
    echo ">>> 已检出 wuji-retargeting，但未检出 hug；已装 grasp extras，HUG 需另跑 ./init.sh hug-env"
  fi

  if [[ ${#skipped[@]} -gt 0 ]]; then
    echo ">>> 以下已检出子模块暂无统一 pip 安装，已跳过："
    local s
    for s in "${skipped[@]}"; do
      echo "    - $s"
    done
  fi

  if [[ "$actions" -eq 0 && ${#skipped[@]} -eq 0 ]]; then
    echo ">>> 没有已检出的可安装子模块。请先 ./init.sh submodules"
    return 1
  fi

  echo ">>> install-checked 完成。"
}

install_projects() {
  install_motion_projects
}

setup_hug_env() {
  local hug_dir="$ROOT_DIR/submodules/hug"
  local setup_script="$hug_dir/scripts/setup_uv.sh"

  if [[ ! -d "$hug_dir" ]]; then
    echo "未找到 $hug_dir 。请先 ./init.sh submodules"
    exit 1
  fi
  if [[ ! -f "$setup_script" ]]; then
    echo "未找到 $setup_script（HUG 子模块未检出完整）。请先 ./init.sh submodules"
    exit 1
  fi
  echo ">>> 配置 HUG 独立环境（Python 3.10，不混入父仓 .venv）"
  bash "$setup_script"
  echo ">>> HUG 环境完成。激活: source $hug_dir/.venv/bin/activate"
  echo ">>> 父仓 grasp-generation / ros2-stack / motion-generation 仍用父仓环境，不要混用。"
  if [[ ! -f "$hug_dir/checkpoints/hug_full.safetensors" ]]; then
    echo ">>> 未找到 checkpoints/hug_full.safetensors。"
    echo "    source $hug_dir/.venv/bin/activate && bash scripts/download_assets.sh"
  fi
  if [[ ! -f "$hug_dir/assets/mano_models/models/MANO_RIGHT.pkl" ]]; then
    echo ">>> 未找到 MANO_RIGHT.pkl。该模型有许可，git 与脚本都不会下载。"
    echo "    1) 注册 https://mano.is.tue.mpg.de/"
    echo "    2) 解压后保证存在: $hug_dir/assets/mano_models/models/MANO_RIGHT.pkl"
    echo "    布局见 submodules/hug/docs/SETUP_LOCAL.md"
  fi
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
    echo ">>> 这会升到 NumPy 2。系统 cv_bridge 不能在这个环境里采头相机。"
    echo "    编排和灵巧手抓取请只用 all-motion / all-grasp，不要和本步骤共用同一个 .venv。"
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
  echo ">>> 使用 backend=$LR_ENV_BACKEND，安装任务编排（interface + composer）"
  install_motion_projects
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

install_miniconda() {
  local miniconda_dir="${HOME}/miniconda3"
  local installer="${miniconda_dir}/miniconda.sh"
  local conda_bin="${miniconda_dir}/bin/conda"

  if [[ -x "$conda_bin" ]]; then
    echo ">>> Miniconda 已安装: $miniconda_dir ($("$conda_bin" --version))"
  elif command -v conda >/dev/null 2>&1; then
    echo ">>> 已检测到 conda: $(command -v conda)（非 ~/miniconda3），跳过 Miniconda 安装。"
    return 0
  else
    echo ">>> 安装 Miniconda 到 $miniconda_dir ..."

    if ! command -v wget >/dev/null 2>&1; then
      echo "未检测到 wget，请先安装 wget。"
      exit 1
    fi

    mkdir -p "$miniconda_dir"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
    bash "$installer" -b -u -p "$miniconda_dir"
    rm -f "$installer"
    echo ">>> Miniconda 安装完成: $miniconda_dir"
  fi

  export PATH="${miniconda_dir}/bin:${PATH}"

  echo ">>> 配置 conda：初始化 shell 并关闭 base 自动激活..."
  # shellcheck disable=SC1091
  source "${miniconda_dir}/bin/activate"
  conda init --all
  conda config --set auto_activate_base False
  echo ">>> conda 配置完成（auto_activate_base=False）。"
  echo ">>> 若当前 shell 找不到 conda，请重新打开终端。"
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo ">>> uv 已安装: $(command -v uv) ($(uv --version))"
    return 0
  fi

  echo ">>> 安装 uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # 安装脚本默认写入 ~/.local/bin
  if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  if command -v uv >/dev/null 2>&1; then
    echo ">>> uv 安装完成: $(command -v uv) ($(uv --version))"
  else
    echo ">>> uv 安装脚本已执行，但未在当前 PATH 中找到 uv。"
    echo ">>> 请重新打开终端，或将 ~/.local/bin 加入 PATH。"
    exit 1
  fi
}

configure_uv_mirror() {
  local uv_config_dir="$HOME/.config/uv"
  local uv_config_file="$uv_config_dir/uv.toml"

  mkdir -p "$uv_config_dir"

  if [[ -f "$uv_config_file" ]]; then
    cp "$uv_config_file" "$uv_config_file.bak.$(date +%Y%m%d%H%M%S)"
    echo ">>> 已备份现有配置: $uv_config_file.bak.<timestamp>"
  fi

  cat > "$uv_config_file" <<'EOF'
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
EOF

  echo ">>> 已配置 uv PyPI 镜像为清华: https://pypi.tuna.tsinghua.edu.cn/simple"
  echo ">>> 配置文件: $uv_config_file"
}

run_all_motion() {
  local python_version="${1:-}"
  init_submodules_core
  create_env_for_configured_backend "$python_version"
  install_motion_stack
  echo ">>> all-motion 完成。未安装灵巧手。需要时: ./init.sh all-grasp"
}

run_all_grasp() {
  local python_version="${1:-}"
  init_submodules_core
  init_submodules_dex
  create_env_for_configured_backend "$python_version"
  install_motion_stack
  install_grasp_extras
  setup_hug_env
  doctor_modules || true
}

run_all_full() {
  local python_version="${1:-}"
  run_all_motion "$python_version"
  install_lerobot_stack
}

doctor_modules() {
  local missing=0
  local py=""
  local hug_dir="$ROOT_DIR/submodules/hug"
  local iface_dir="$ROOT_DIR/submodules/ros2_robot_interface"
  local rac_dir="$ROOT_DIR/$ROBOT_ACTION_COMPOSER_SUBMODULE_PATH"
  local wuji_dir="$ROOT_DIR/submodules/wuji-retargeting"

  _doc_ok() { echo "  [ok] $*"; }
  _doc_miss() { echo "  [缺失] $*"; missing=1; }

  lr_env_load_config "$ROOT_DIR"
  echo ">>> 模块检查（backend=$LR_ENV_BACKEND）。缺灵巧手时，任务编排仍可使用。"

  if git -C "$iface_dir" rev-parse --git-dir >/dev/null 2>&1; then
    _doc_ok "子模块 ros2_robot_interface"
  else
    _doc_miss "子模块 ros2_robot_interface → ./init.sh submodules-core"
  fi
  if git -C "$rac_dir" rev-parse --git-dir >/dev/null 2>&1; then
    _doc_ok "子模块 robot_action_composer"
  else
    _doc_miss "子模块 robot_action_composer → ./init.sh submodules-core"
  fi
  if git -C "$hug_dir" rev-parse --git-dir >/dev/null 2>&1; then
    _doc_ok "子模块 hug（git；权重和 MANO 另计）"
  else
    _doc_miss "子模块 hug → ./init.sh submodules-dex （不做灵巧手可忽略）"
  fi
  if git -C "$wuji_dir" rev-parse --git-dir >/dev/null 2>&1; then
    _doc_ok "子模块 wuji-retargeting（参考仓；运行时用 wuji-sdk）"
  else
    _doc_miss "子模块 wuji-retargeting → ./init.sh submodules-dex （不做灵巧手可忽略）"
  fi

  if [[ "$LR_ENV_BACKEND" == "uv" ]]; then
    py="$(lr_env_uv_venv_path)/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    py="$(conda run -n "$LR_ENV_CONDA_NAME" which python 2>/dev/null || true)"
  fi

  if [[ -n "$py" && -x "$py" ]]; then
    _doc_ok "父仓解释器 $py"
  else
    _doc_miss "父仓环境 → ./init.sh env"
    py=""
  fi

  if [[ -n "$py" ]]; then
    if "$py" -c "import importlib.metadata as m; m.version('robot-action-composer'); m.version('ros2-robot-interface')" >/dev/null 2>&1; then
      _doc_ok "已安装 interface + composer（motion-generation / ros2-stack）"
      if ! "$py" -c "import ros2_robot_interface, robot_action_composer" >/dev/null 2>&1; then
        _doc_miss "包已安装，但当前解释器导入失败。常见原因是没 source 到含自定义消息的 ROS 工作空间（例如 arms_ros2_control_msgs）。处理: ./init.sh ros2-workspace，或先 source 该工作空间再开终端"
      fi
    else
      _doc_miss "interface 或 composer 未装进当前环境 → ./init.sh install"
    fi
    if "$py" -c "import viser" >/dev/null 2>&1; then
      _doc_ok "viser（抓取界面）"
    else
      _doc_miss "viser → ./init.sh install-grasp （只做任务编排可忽略）"
    fi
    if "$py" -c "import wuji_sdk" >/dev/null 2>&1; then
      _doc_ok "wuji-sdk（参考手型 retarget；其它 hand_profile 不一定需要）"
    else
      _doc_miss "wuji-sdk → ./init.sh install-grasp （当前参考手型需要；其它手型可忽略）"
    fi
    if "$py" -c "import rclpy" >/dev/null 2>&1; then
      _doc_ok "rclpy"
    else
      _doc_miss "rclpy。source ROS 2，或 ./init.sh ros2-workspace"
    fi
    np_ver="$("$py" -c "import numpy; print(numpy.__version__)" 2>/dev/null || true)"
    if [[ "$np_ver" == 2.* ]]; then
      _doc_miss "numpy $np_ver 与系统 cv_bridge（NumPy 1.x）冲突，头相机采图会崩。编排/抓取环境执行: uv pip install 'numpy>=1.26,<2'（不要用 install-lerobot 的那个环境）"
    elif [[ -n "$np_ver" ]]; then
      _doc_ok "numpy $np_ver（<2，可与 cv_bridge 一起采图）"
    fi
  fi

  if [[ -x "$hug_dir/.venv/bin/python" ]]; then
    _doc_ok "HUG 独立 venv $hug_dir/.venv"
  else
    _doc_miss "HUG venv → ./init.sh hug-env （不做 HUG 推理可忽略）"
  fi
  if [[ -f "$hug_dir/checkpoints/hug_full.safetensors" ]]; then
    _doc_ok "HUG 权重 checkpoints/hug_full.safetensors"
  else
    _doc_miss "HUG 权重未进 git。hug-env 之后: bash submodules/hug/scripts/download_assets.sh"
  fi
  if [[ -f "$hug_dir/assets/mano_models/models/MANO_RIGHT.pkl" ]]; then
    _doc_ok "MANO_RIGHT.pkl"
  else
    _doc_miss "MANO_RIGHT.pkl 需注册下载 https://mano.is.tue.mpg.de/ → assets/mano_models/models/"
  fi

  echo
  if [[ "$missing" -eq 0 ]]; then
    echo ">>> 已检查的项都在。"
  else
    echo ">>> 有缺失项。只做任务编排时，忽略 hug / viser / wuji-sdk / MANO 那几行即可。"
  fi
  return 0
}

MENU_IDS=()
MENU_LABELS=()
MENU_KINDS=()
MENU_N=0

menu_add() {
  MENU_IDS+=("$1")
  MENU_LABELS+=("$2")
  MENU_KINDS+=("item")
  MENU_N=$((MENU_N + 1))
}

menu_add_header() {
  MENU_IDS+=("")
  MENU_LABELS+=("$1")
  MENU_KINDS+=("header")
  MENU_N=$((MENU_N + 1))
}

menu_is_selectable() {
  [[ "${MENU_KINDS[$1]}" == "item" ]]
}

menu_next_selectable() {
  local start="$1"
  local step="$2"
  local idx="$start"
  local i
  for ((i = 0; i < MENU_N; i++)); do
    idx=$(( (idx + step + MENU_N) % MENU_N ))
    if menu_is_selectable "$idx"; then
      echo "$idx"
      return 0
    fi
  done
  echo "$start"
}

menu_first_selectable() {
  local i
  for ((i = 0; i < MENU_N; i++)); do
    if menu_is_selectable "$i"; then
      echo "$i"
      return 0
    fi
  done
  echo 0
}

build_main_menu() {
  MENU_IDS=()
  MENU_LABELS=()
  MENU_KINDS=()
  MENU_N=0

  menu_add_header "[子模块]"
  menu_add "submodules" "勾选并初始化子模块（默认 core；可加 dex / roboapi-fasim 等）"
  menu_add "update-submodules" "更新已检出子模块到最新 main"
  menu_add "install-checked" "安装已检出子模块对应依赖（core / roboapi-fasim / dex）"

  menu_add_header "[环境与安装]"
  menu_add "env" "按当前 backend 创建环境"
  menu_add "install" "安装任务编排（interface + composer；不含灵巧手）"
  menu_add "install-grasp" "安装抓取 extras（viser + wuji-sdk，可选）"
  menu_add "install-lerobot" "安装 lerobot 相关（PyTorch + lerobot + 插件）"
  menu_add "hug-env" "配置 HUG 独立 venv（Python 3.10，可选）"
  menu_add "doctor" "检查各模块是否就绪"

  menu_add_header "[一键执行]"
  menu_add "all-motion" "任务编排（core + 环境 + 安装；不含灵巧手）"
  menu_add "all" "任务编排 + lerobot"
  menu_add "all-grasp" "任务编排 + 抓取 extras + HUG"

  menu_add_header "[配置]"
  if [[ "$LR_ENV_BACKEND" == "conda" ]]; then
    menu_add "install-miniconda" "安装 Miniconda"
    menu_add "pypi-mirror" "配置 NJU PyPI 镜像（pip）"
  else
    menu_add "install-uv" "安装 uv"
    menu_add "uv-mirror" "配置 uv PyPI 镜像（清华）"
  fi
  menu_add "ros2-workspace" "配置 ROS2 工作空间"
  menu_add "set-backend" "切换 backend (conda/uv)"
  if [[ "$LR_ENV_BACKEND" == "conda" ]]; then
    menu_add "conda-runtime" "配置 conda 运行时库"
  fi

  menu_add_header "[其他]"
  menu_add "quit" "退出"
}

draw_main_menu() {
  local cursor_idx="$1"
  local i cursor
  for ((i = 0; i < MENU_N; i++)); do
    if [[ "${MENU_KINDS[$i]}" == "header" ]]; then
      if [[ "$i" -gt 0 ]]; then
        printf "\n"
      fi
      printf "  %s\n" "${MENU_LABELS[$i]}"
      continue
    fi
    if [[ "$i" -eq "$cursor_idx" ]]; then
      cursor=">"
      printf " %s %s\n" "$cursor" "${MENU_LABELS[$i]}"
    else
      printf "   %s\n" "${MENU_LABELS[$i]}"
    fi
  done
}

menu_draw_line_count() {
  local i count=0
  for ((i = 0; i < MENU_N; i++)); do
    if [[ "${MENU_KINDS[$i]}" == "header" ]]; then
      if [[ "$i" -gt 0 ]]; then
        count=$((count + 1))
      fi
      count=$((count + 1))
    else
      count=$((count + 1))
    fi
  done
  echo "$count"
}

pick_main_menu_tty() {
  local idx key k2 k3 total
  local old_stty
  idx="$(menu_first_selectable)"
  old_stty=$(stty -g)
  restore_menu_picker() {
    stty "$old_stty" 2>/dev/null || true
    printf '\033[?25h'
  }
  trap restore_menu_picker EXIT INT TERM
  stty -echo -icanon
  printf '\033[?25l'

  echo "请选择要执行的操作 (backend=$LR_ENV_BACKEND)"
  echo "  ↑/↓ 移动  Enter 确认  q 退出"
  echo "  提示: 编排与 lerobot 不要混同一环境；抓取保持 numpy<2"
  echo ""
  total=$((4 + $(menu_draw_line_count)))
  draw_main_menu "$idx"

  while true; do
    IFS= read -rsn1 key || key=""
    if [[ "$key" == $'\x1b' ]]; then
      IFS= read -rsn1 -t 0.05 k2 || k2=""
      IFS= read -rsn1 -t 0.05 k3 || k3=""
      if [[ "$k3" == "A" || "$k2" == "A" ]]; then
        idx="$(menu_next_selectable "$idx" -1)"
      elif [[ "$k3" == "B" || "$k2" == "B" ]]; then
        idx="$(menu_next_selectable "$idx" 1)"
      fi
    elif [[ "$key" == "q" || "$key" == "Q" ]]; then
      restore_menu_picker
      trap - EXIT INT TERM
      echo ""
      MENU_CHOICE="quit"
      return 0
    elif [[ -z "$key" ]]; then
      restore_menu_picker
      trap - EXIT INT TERM
      echo ""
      MENU_CHOICE="${MENU_IDS[$idx]}"
      return 0
    fi
    printf '\033[%sA' "$total"
    echo "请选择要执行的操作 (backend=$LR_ENV_BACKEND)"
    echo "  ↑/↓ 移动  Enter 确认  q 退出"
    echo "  提示: 编排与 lerobot 不要混同一环境；抓取保持 numpy<2"
    echo ""
    draw_main_menu "$idx"
  done
}

pick_main_menu_notty() {
  local i input n=0
  local ids=() labels=()
  echo "请选择要执行的操作 (backend=$LR_ENV_BACKEND，非交互终端)"
  echo "  提示: 编排与 lerobot 不要混同一环境；抓取保持 numpy<2"
  echo ""
  for ((i = 0; i < MENU_N; i++)); do
    if [[ "${MENU_KINDS[$i]}" == "header" ]]; then
      if [[ "$i" -gt 0 ]]; then
        echo ""
      fi
      printf "  %s\n" "${MENU_LABELS[$i]}"
      continue
    fi
    n=$((n + 1))
    ids+=("${MENU_IDS[$i]}")
    labels+=("${MENU_LABELS[$i]}")
    printf "    %2d) %s\n" "$n" "${MENU_LABELS[$i]}"
  done
  echo ""
  read -r -p "输入序号: " input
  if [[ "$input" =~ ^[0-9]+$ ]] && [[ "$input" -ge 1 ]] && [[ "$input" -le "$n" ]]; then
    MENU_CHOICE="${ids[$((input - 1))]}"
    return 0
  fi
  echo "无效选项。"
  return 1
}

run_main_menu_choice() {
  local choice="$1"
  local input_python_version backend_choice
  case "$choice" in
    submodules)
      init_submodules_interactive
      ;;
    update-submodules)
      update_submodules_to_main
      ;;
    install-checked)
      install_checked_submodules
      ;;
    env)
      read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
      create_env_for_configured_backend "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
      ;;
    install)
      install_motion_stack
      ;;
    install-grasp)
      install_grasp_extras
      ;;
    install-lerobot)
      install_lerobot_stack
      ;;
    hug-env)
      setup_hug_env
      ;;
    doctor)
      doctor_modules
      ;;
    all-grasp)
      read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
      run_all_grasp "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
      ;;
    all-motion)
      read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
      run_all_motion "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
      ;;
    all)
      read -r -p "输入 Python 版本（默认 $DEFAULT_PYTHON_VERSION）: " input_python_version
      run_all_full "${input_python_version:-$DEFAULT_PYTHON_VERSION}"
      ;;
    install-miniconda)
      install_miniconda
      ;;
    install-uv)
      install_uv
      ;;
    pypi-mirror)
      configure_nju_pypi_mirror
      ;;
    uv-mirror)
      configure_uv_mirror
      ;;
    ros2-workspace)
      configure_ros2_workspace_source
      ;;
    set-backend)
      read -r -p "输入 backend [conda/uv]（当前 $LR_ENV_BACKEND）: " backend_choice
      backend_choice="${backend_choice:-$LR_ENV_BACKEND}"
      lr_env_set_backend "$backend_choice"
      ;;
    conda-runtime)
      configure_conda_runtime_libs
      ;;
    quit)
      echo "已退出。"
      ;;
    *)
      echo "无效选项: $choice"
      return 1
      ;;
  esac
}

interactive_menu() {
  local MENU_CHOICE=""
  lr_env_load_config "$ROOT_DIR"
  build_main_menu
  if [[ -t 0 && -t 1 ]]; then
    pick_main_menu_tty || return 1
  else
    pick_main_menu_notty || return 1
  fi
  run_main_menu_choice "$MENU_CHOICE"
}

main() {
  case "${1:-}" in
    submodules)
      run_submodules_command "${@:2}"
      ;;
    submodules-core)
      init_submodules_core
      ;;
    submodules-dex)
      init_submodules_dex
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
    install-checked)
      parse_install_backend_args "${@:2}"
      install_checked_submodules
      ;;
    install-grasp)
      parse_install_backend_args "${@:2}"
      install_grasp_extras
      ;;
    install-plugins|install-lerobot)
      parse_install_backend_args "${@:2}"
      install_lerobot_stack
      ;;
    install-uv)
      install_uv
      ;;
    install-miniconda)
      install_miniconda
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
    uv-mirror)
      configure_uv_mirror
      ;;
    hug-env)
      setup_hug_env
      ;;
    doctor)
      doctor_modules
      ;;
    all-motion)
      run_all_motion "${2:-}"
      ;;
    all-grasp)
      run_all_grasp "${2:-}"
      ;;
    all)
      run_all_full "${2:-}"
      ;;
    "")
      interactive_menu
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
