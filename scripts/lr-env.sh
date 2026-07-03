#!/usr/bin/env bash
# lerobot_ros2 环境配置与激活（由 init.sh source）

LR_ENV_DEFAULT_BACKEND="uv"
LR_ENV_DEFAULT_CONDA_NAME="lerobot-ros2"
LR_ENV_DEFAULT_UV_VENV=".venv"
LR_ENV_DEFAULT_LEROBOT_VERSION="0.5.1"

LR_ENV_BACKEND_OVERRIDE="${LR_ENV_BACKEND_OVERRIDE:-}"
LR_ENV_BACKEND=""
LR_ENV_CONDA_NAME=""
LR_ENV_UV_VENV=""
LR_ENV_ROS2_WORKSPACE=""
LR_ENV_LEROBOT_VERSION=""

lr_env_root_dir() {
  if [[ -n "${LR_ENV_ROOT_DIR:-}" ]]; then
    echo "$LR_ENV_ROOT_DIR"
    return 0
  fi
  echo "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
}

lr_env_config_paths() {
  local root
  root="$(lr_env_root_dir)"
  echo "${root}/.fa-env.local.toml"
  echo "${root}/.fa-env.toml"
}

lr_env_primary_config_file() {
  local path
  while IFS= read -r path; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done < <(lr_env_config_paths)
  echo ""
}

lr_env_ensure_config_file() {
  local root cfg
  root="$(lr_env_root_dir)"
  cfg="${root}/.fa-env.toml"
  if [[ -f "$cfg" ]]; then
    return 0
  fi
  cat > "$cfg" <<'EOF'
# lerobot_ros2 运行时环境配置

backend = "uv"

[conda]
name = "lerobot-ros2"

[uv]
venv = ".venv"

[lerobot]
version = "0.5.1"

[ros2]
# workspace = "~/ros2_ws"
EOF
}

lr_env_toml_get() {
  local file="$1"
  local section="${2:-}"
  local key="$3"
  local current_section=""
  local line k v

  [[ -f "$file" ]] || return 1

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "$line" ]] || continue

    if [[ "$line" =~ ^\[(.+)\]$ ]]; then
      current_section="${BASH_REMATCH[1]}"
      continue
    fi

    if [[ "$line" != *"="* ]]; then
      continue
    fi

    k="$(echo "${line%%=*}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    v="$(echo "${line#*=}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')"

    if [[ "$current_section" == "$section" && "$k" == "$key" && -n "$v" ]]; then
      echo "$v"
      return 0
    fi
  done < "$file"
  return 1
}

lr_env_expand_path() {
  local p="$1"
  local root
  root="$(lr_env_root_dir)"
  p="${p/#\~/$HOME}"
  if [[ "$p" != /* ]]; then
    echo "${root}/${p}"
  else
    echo "$p"
  fi
}

lr_env_python_is_conda() {
  local py_path="$1"
  [[ "$py_path" == *miniconda* || "$py_path" == *anaconda* || "$py_path" == *conda* ]] && return 0
  return 1
}

lr_env_python_version_matches() {
  local py_path="$1"
  local want_version="$2"
  local actual=""
  [[ -x "$py_path" ]] || return 1
  actual="$("$py_path" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  [[ "$actual" == "$want_version" ]]
}

lr_env_resolve_system_python() {
  local want_version="${1:-3.12}"
  local candidate resolved
  local -a candidates=()

  candidates=(
    "/usr/bin/python${want_version}"
    "/usr/local/bin/python${want_version}"
  )

  local ros_distro="${ROS_DISTRO:-jazzy}"
  if [[ -x "/opt/ros/${ros_distro}/bin/python3" ]]; then
    candidates+=("/opt/ros/${ros_distro}/bin/python3")
  fi

  if command -v "python${want_version}" >/dev/null 2>&1; then
    candidates+=("$(command -v "python${want_version}")")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    resolved="$(readlink -f "$candidate" 2>/dev/null || echo "$candidate")"
    lr_env_python_is_conda "$resolved" && continue
    lr_env_python_version_matches "$resolved" "$want_version" || continue
    echo "$resolved"
    return 0
  done

  return 1
}

lr_env_create_uv_venv() {
  local want_version="$1"
  local venv_path="$2"
  local python_exe=""

  if python_exe="$(lr_env_resolve_system_python "$want_version")"; then
    echo ">>> 使用系统解释器: $python_exe"
    uv venv --python "$python_exe" --system-site-packages "$venv_path"
    return 0
  fi

  echo ">>> WARN: 未找到 Python ${want_version} 的系统解释器，回退为 uv 托管的 Python ${want_version}"
  uv venv --python "$want_version" --system-site-packages "$venv_path"
}

lr_env_load_config() {
  local root="${1:-}"
  local cfg=""
  local file_val=""
  local user_backend="${LR_ENV_BACKEND_OVERRIDE:-}"

  if [[ -n "$root" ]]; then
    LR_ENV_ROOT_DIR="$root"
  fi
  root="$(lr_env_root_dir)"

  LR_ENV_BACKEND="$LR_ENV_DEFAULT_BACKEND"
  LR_ENV_CONDA_NAME="$LR_ENV_DEFAULT_CONDA_NAME"
  LR_ENV_UV_VENV="$LR_ENV_DEFAULT_UV_VENV"
  LR_ENV_ROS2_WORKSPACE=""
  LR_ENV_LEROBOT_VERSION="$LR_ENV_DEFAULT_LEROBOT_VERSION"

  lr_env_ensure_config_file
  cfg="$(lr_env_primary_config_file)"

  if [[ -n "$cfg" ]]; then
    file_val="$(lr_env_toml_get "$cfg" "" "backend" 2>/dev/null || true)"
    [[ -n "$file_val" ]] && LR_ENV_BACKEND="$file_val"

    file_val="$(lr_env_toml_get "$cfg" "conda" "name" 2>/dev/null || true)"
    [[ -n "$file_val" ]] && LR_ENV_CONDA_NAME="$file_val"

    file_val="$(lr_env_toml_get "$cfg" "uv" "venv" 2>/dev/null || true)"
    [[ -n "$file_val" ]] && LR_ENV_UV_VENV="$file_val"

    file_val="$(lr_env_toml_get "$cfg" "ros2" "workspace" 2>/dev/null || true)"
    [[ -n "$file_val" ]] && LR_ENV_ROS2_WORKSPACE="$file_val"

    file_val="$(lr_env_toml_get "$cfg" "lerobot" "version" 2>/dev/null || true)"
    [[ -n "$file_val" ]] && LR_ENV_LEROBOT_VERSION="$file_val"
  fi

  if [[ -n "$user_backend" ]]; then
    LR_ENV_BACKEND="$user_backend"
  fi
}

lr_env_uv_venv_path() {
  lr_env_expand_path "${LR_ENV_UV_VENV:-$LR_ENV_DEFAULT_UV_VENV}"
}

lr_env_set_toml_value() {
  local file="$1"
  local section="$2"
  local key="$3"
  local value="$4"
  local tmp
  local in_target=0
  local replaced=0
  local line current_section=""

  lr_env_ensure_config_file
  [[ -f "$file" ]] || return 1

  if [[ -z "$section" ]]; then
    in_target=1
  fi

  tmp="$(mktemp)"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
      current_section="${BASH_REMATCH[1]}"
      if [[ -z "$section" ]]; then
        in_target=0
      else
        [[ "$current_section" == "$section" ]] && in_target=1 || in_target=0
      fi
      printf '%s\n' "$line" >>"$tmp"
      continue
    fi

    if [[ "$in_target" -eq 1 && "$line" =~ ^[[:space:]]*${key}[[:space:]]*= ]]; then
      printf '%s = "%s"\n' "$key" "$value" >>"$tmp"
      replaced=1
      continue
    fi

    printf '%s\n' "$line" >>"$tmp"
  done <"$file"

  if [[ "$replaced" -eq 0 ]]; then
    if [[ -n "$section" ]]; then
      if ! grep -q "^\[${section}\]" "$file"; then
        printf '\n[%s]\n' "$section" >>"$tmp"
      fi
      printf '%s = "%s"\n' "$key" "$value" >>"$tmp"
    else
      sed -i "1i${key} = \"${value}\"" "$file" 2>/dev/null || {
        printf '%s = "%s"\n' "$key" "$value" | cat - "$file" >"$tmp"
        mv "$tmp" "$file"
        return 0
      }
      return 0
    fi
  fi

  mv "$tmp" "$file"
}

lr_env_set_backend() {
  local new_backend="$1"
  local cfg
  case "$new_backend" in
    conda|uv) ;;
    *)
      echo "backend 必须是 conda 或 uv"
      return 1
      ;;
  esac
  lr_env_ensure_config_file
  cfg="$(lr_env_root_dir)/.fa-env.toml"
  if grep -qE '^[[:space:]]*backend[[:space:]]*=' "$cfg"; then
    sed -i -E "s/^[[:space:]]*backend[[:space:]]*=.*/backend = \"${new_backend}\"/" "$cfg"
  else
    lr_env_set_toml_value "$cfg" "" "backend" "$new_backend"
  fi
  LR_ENV_BACKEND="$new_backend"
  echo ">>> 已设置 backend=$new_backend（$cfg）"
}

lr_env_set_ros2_workspace() {
  local ws_path="$1"
  local cfg
  lr_env_ensure_config_file
  cfg="$(lr_env_root_dir)/.fa-env.toml"
  lr_env_set_toml_value "$cfg" "ros2" "workspace" "$ws_path"
  LR_ENV_ROS2_WORKSPACE="$ws_path"
}

lr_env_is_ros_system_package() {
  local pkg="$1"
  pkg="${pkg%%\[*}"
  case "$pkg" in
    rclpy|ros2-robot-interface|ros2_robot_interface|cv-bridge|cv_bridge) return 0 ;;
    *-msgs) return 0 ;;
  esac
  return 1
}

lr_env_pyproject_pypi_deps() {
  local pyproject="$1"
  local line pkg

  [[ -f "$pyproject" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*,$//;s/^"//;s/"$//')"
    [[ -z "$line" || "$line" == "["* || "$line" == "]"* ]] && continue
    pkg="${line%%[<>=!~;]*}"
    pkg="$(echo "$pkg" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$pkg" ]] && continue
    if lr_env_is_ros_system_package "$pkg"; then
      continue
    fi
    echo "$line"
  done < <(
    awk '
      /^dependencies = \[/ { in_deps=1; next }
      in_deps && /^\]/ { in_deps=0; next }
      in_deps { print }
    ' "$pyproject"
  )
}

lr_env_safe_source_setup() {
  local setup_script="$1"
  local label="${2:-$setup_script}"

  [[ -f "$setup_script" ]] || return 1

  set +u
  # shellcheck disable=SC1090
  source "$setup_script"
  echo ">>> 已 source ROS2: $label"
  return 0
}

lr_env_source_ros2_workspace() {
  local ws_path="${1:-}"
  local setup_script

  [[ -n "$ws_path" ]] || return 0

  ws_path="$(lr_env_expand_path "$ws_path")"
  setup_script="${ws_path}/install/setup.bash"

  if lr_env_safe_source_setup "$setup_script" "$setup_script"; then
    return 0
  fi
  echo ">>> WARN: 未找到 ROS2 setup.bash: $setup_script"
}

lr_env_try_source_ros2() {
  if [[ -n "${LR_ENV_ROS2_WORKSPACE:-}" ]]; then
    lr_env_source_ros2_workspace "$LR_ENV_ROS2_WORKSPACE"
    return 0
  fi
  local distro="${ROS_DISTRO:-jazzy}"
  local setup="/opt/ros/${distro}/setup.bash"
  if lr_env_safe_source_setup "$setup" "$setup"; then
    return 0
  fi
  echo ">>> WARN: 未配置 [ros2].workspace 且未找到 $setup；运行 ROS 节点前请 source ROS2。"
}

lr_env_uv_has_ros2_activate_hook() {
  [[ -f "$(lr_env_uv_venv_path)/bin/lr_ros2_workspace.sh" ]]
}

lr_env_uv_pip_install_editable() {
  local project_dir="$1"
  local pyproject="$project_dir/pyproject.toml"
  local deps=()
  local dep

  echo ">>> [uv] editable: $project_dir"
  uv pip install -e "$project_dir" --no-deps

  while IFS= read -r dep || [[ -n "$dep" ]]; do
    [[ -n "$dep" ]] && deps+=("$dep")
  done < <(lr_env_pyproject_pypi_deps "$pyproject")

  if [[ ${#deps[@]} -gt 0 ]]; then
    echo ">>> [uv] PyPI 依赖: ${deps[*]}"
    uv pip install "${deps[@]}"
  fi
}

lr_env_pip_install_editable() {
  local project_dir="$1"
  local pyproject="$project_dir/pyproject.toml"
  local deps=()
  local dep

  echo ">>> editable: $project_dir"
  python -m pip install -e "$project_dir" --no-deps

  while IFS= read -r dep || [[ -n "$dep" ]]; do
    [[ -n "$dep" ]] && deps+=("$dep")
  done < <(lr_env_pyproject_pypi_deps "$pyproject")

  if [[ ${#deps[@]} -gt 0 ]]; then
    echo ">>> PyPI 依赖: ${deps[*]}"
    python -m pip install "${deps[@]}"
  fi
}

lr_env_write_conda_ros2_hook() {
  local env_prefix="$1"
  local ws_path="$2"
  local activate_dir activate_script

  [[ -n "$env_prefix" && -d "$env_prefix" ]] || return 0

  ws_path="$(lr_env_expand_path "$ws_path")"
  activate_dir="${env_prefix}/etc/conda/activate.d"
  activate_script="${activate_dir}/lr_ros2_workspace.sh"
  mkdir -p "$activate_dir"

  cat >"$activate_script" <<EOF
#!/usr/bin/env bash
if [ -f "${ws_path}/install/setup.bash" ]; then
    set +u
    # shellcheck disable=SC1091
    source "${ws_path}/install/setup.bash"
    echo "[conda activate] Sourced ROS2 workspace: ${ws_path}/install/setup.bash"
else
    echo "[conda activate] WARN: ROS2 setup.bash not found at ${ws_path}/install/setup.bash"
fi
EOF
  chmod +x "$activate_script"
  echo ">>> conda activate.d: $activate_script"
}

lr_env_write_uv_ros2_hook() {
  local venv_path="$1"
  local ws_path="$2"
  local hook_script activate_file
  local marker_begin="# >>> lerobot_ros2 ros2 >>>"
  local marker_end="# <<< lerobot_ros2 ros2 <<<"

  [[ -f "${venv_path}/bin/activate" ]] || return 0

  ws_path="$(lr_env_expand_path "$ws_path")"
  hook_script="${venv_path}/bin/lr_ros2_workspace.sh"
  activate_file="${venv_path}/bin/activate"

  cat >"$hook_script" <<EOF
#!/usr/bin/env bash
# 由 ./init.sh ros2-workspace 生成，source .venv/bin/activate 时自动执行
_LR_ROS2_WS="${ws_path}"
if [[ -f "\${_LR_ROS2_WS}/install/setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "\${_LR_ROS2_WS}/install/setup.bash"
    echo "[venv activate] Sourced ROS2 workspace: \${_LR_ROS2_WS}/install/setup.bash"
elif [[ -f "/opt/ros/\${ROS_DISTRO:-jazzy}/setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "/opt/ros/\${ROS_DISTRO:-jazzy}/setup.bash"
    echo "[venv activate] Sourced system ROS2: /opt/ros/\${ROS_DISTRO:-jazzy}"
else
    echo "[venv activate] WARN: 未找到 ROS2 setup.bash（工作空间或 /opt/ros）"
fi
EOF
  chmod +x "$hook_script"

  if grep -qF "$marker_begin" "$activate_file" 2>/dev/null; then
    awk -v begin="$marker_begin" -v end="$marker_end" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "$activate_file" >"${activate_file}.tmp"
    mv "${activate_file}.tmp" "$activate_file"
  fi

  cat >>"$activate_file" <<EOF

$marker_begin
if [[ -n "\${VIRTUAL_ENV:-}" && -f "\${VIRTUAL_ENV}/bin/lr_ros2_workspace.sh" ]]; then
    # shellcheck disable=SC1091
    . "\${VIRTUAL_ENV}/bin/lr_ros2_workspace.sh"
fi
$marker_end
EOF

  echo ">>> venv activate 挂钩: $activate_file"
  echo ">>> ROS2 钩子脚本: $hook_script"
}

lr_env_conda_env_prefix() {
  local env_name="${1:-$LR_ENV_CONDA_NAME}"
  if ! command -v conda >/dev/null 2>&1; then
    return 1
  fi
  conda env list | awk -v name="$env_name" '$1==name {print $(NF); found=1} END { exit !found }'
}

lr_env_apply_ros2_hooks() {
  local ws_path="${1:-}"
  local apply_all="${2:-0}"
  local venv_path env_prefix
  local do_uv=0 do_conda=0

  lr_env_load_config "$(lr_env_root_dir)"
  [[ -n "$ws_path" ]] || ws_path="$LR_ENV_ROS2_WORKSPACE"
  [[ -n "$ws_path" ]] || return 0

  if [[ "$apply_all" == "1" ]]; then
    do_uv=1
    do_conda=1
  else
    case "$LR_ENV_BACKEND" in
      uv) do_uv=1 ;;
      conda) do_conda=1 ;;
    esac
  fi

  if [[ "$do_uv" -eq 1 ]]; then
    venv_path="$(lr_env_uv_venv_path)"
    if [[ -f "${venv_path}/bin/activate" ]]; then
      lr_env_write_uv_ros2_hook "$venv_path" "$ws_path"
    else
      echo ">>> WARN: 未找到 $venv_path，跳过 uv 挂钩（请先 ./init.sh env）"
    fi
  fi

  if [[ "$do_conda" -eq 1 ]]; then
    if env_prefix="$(lr_env_conda_env_prefix "$LR_ENV_CONDA_NAME")"; then
      lr_env_write_conda_ros2_hook "$env_prefix" "$ws_path"
    else
      echo ">>> WARN: conda 环境 '$LR_ENV_CONDA_NAME' 不存在，跳过 conda 挂钩（请先 ./init.sh env）"
    fi
  fi
}

lr_env_activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "未检测到 conda，请先安装并配置 conda。"
    return 1
  fi

  if ! conda env list | awk '{print $1}' | grep -Fxq "$LR_ENV_CONDA_NAME"; then
    echo "conda 环境 '$LR_ENV_CONDA_NAME' 不存在，请先执行: ./init.sh env"
    return 1
  fi

  set +u
  eval "$(conda shell.bash hook)"
  conda activate "$LR_ENV_CONDA_NAME"
  return 0
}

lr_env_activate_uv() {
  local venv_path
  venv_path="$(lr_env_uv_venv_path)"

  if [[ ! -f "${venv_path}/bin/activate" ]]; then
    echo "uv 虚拟环境不存在: $venv_path"
    echo "请先执行: ./init.sh env"
    return 1
  fi

  set +u
  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate"
  return 0
}

lr_env_activate() {
  local root="${1:-}"

  lr_env_load_config "$root"

  case "$LR_ENV_BACKEND" in
    conda)
      lr_env_activate_conda || return 1
      ;;
    uv)
      lr_env_activate_uv || return 1
      ;;
    *)
      echo "未知 backend: $LR_ENV_BACKEND（应为 conda 或 uv）"
      return 1
      ;;
  esac

  case "$LR_ENV_BACKEND" in
    uv)
      if ! lr_env_uv_has_ros2_activate_hook; then
        lr_env_try_source_ros2
      fi
      ;;
    conda)
      if [[ -n "$LR_ENV_ROS2_WORKSPACE" ]]; then
        lr_env_source_ros2_workspace "$LR_ENV_ROS2_WORKSPACE"
      fi
      ;;
  esac

  echo ">>> 已激活环境 (backend=$LR_ENV_BACKEND)"
  return 0
}

lr_env_install_editable() {
  local project_dir="$1"
  case "$LR_ENV_BACKEND" in
    uv)
      if ! command -v uv >/dev/null 2>&1; then
        echo "未检测到 uv，请先安装: https://docs.astral.sh/uv/"
        return 1
      fi
      echo ">>> [uv] rclpy / *-msgs / cv_bridge 由系统 ROS2 提供，不从 PyPI 解析"
      lr_env_uv_pip_install_editable "$project_dir"
      ;;
    conda|*)
      lr_env_pip_install_editable "$project_dir"
      ;;
  esac
}

lr_env_pip_install() {
  local pkg="$1"
  case "$LR_ENV_BACKEND" in
    uv) uv pip install "$pkg" ;;
    conda|*) python -m pip install "$pkg" ;;
  esac
}

lr_env_pip_install_torch() {
  local index_url="https://download.pytorch.org/whl/cu128"
  case "$LR_ENV_BACKEND" in
    uv)
      uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$index_url"
      ;;
    conda|*)
      python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url "$index_url"
      ;;
  esac
}
