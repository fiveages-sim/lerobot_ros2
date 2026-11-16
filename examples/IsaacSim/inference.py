#!/usr/bin/env python3
"""Unified IsaacSim inference launcher (LeRobot + ROS2).

Robot / task discovery and ``pick``/``place`` flattening live in ``robot_action_composer``
(``discovery.registry_loader``, ``task_config_io``) — same as motion / 录制入口。
策略循环与 ROS2 发送在 ``online_infer/core.py``。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified IsaacSim inference launcher")
    parser.add_argument("--robot", type=str, default=None, help="Robot key (e.g. dobot_cr5)")
    parser.add_argument("--task", type=str, default=None, help="Task key (e.g. pick_place)")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root: examples/IsaacSim (robots/) or a fa-project-usd project dir (env/)",
    )
    parser.add_argument(
        "--robots-root",
        type=str,
        default=None,
        help="Shared robot profile root (default: <workspace>/robots or the scene robot.yaml robots_root)",
    )
    parser.add_argument(
        "--object-prim",
        dest="object_prim",
        type=str,
        default="",
        help="（已废弃，保留兼容）推理脚本不再复位仿真 —— 开关/复位场景由人负责。"
             "给了也只打印，不触发任何仿真写操作。",
    )
    parser.add_argument(
        "--no-reset",
        dest="no_reset",
        action="store_true",
        help="（已废弃，保留兼容）推理脚本现在**始终**不复位仿真，此开关无作用。",
    )
    parser.add_argument(
        "--n-action-steps",
        dest="n_action_steps",
        type=int,
        default=100,
        help="一次预测后连续执行多少帧再回读观测（覆盖 checkpoint 里的值）。"
             "**必须 ≥ 策略从当前观测到关键动作的帧数**：本任务策略第 49 帧才合爪，"
             "设 25 时那一步不在队列里，手臂只走到 2.06 就重算 → 反复够不到；"
             "设 100（=chunk_size）可一把完成抓取+交接+插管（已实测成功）。"
             "默认 100；设 0 = 用 checkpoint 原值（通常 1，会死锁）。",
    )
    launcher_args, core_args = parser.parse_known_args()
    # 透传给 core.py（它自己也要 parse，所以必须显式追加）
    if launcher_args.n_action_steps is not None:
        core_args += ["--n-action-steps", str(launcher_args.n_action_steps)]
    return launcher_args, core_args

def _iter_queue_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(block: Any) -> None:
        if not isinstance(block, dict):
            return
        if "parallel" in block:
            for sub in block.get("parallel", []):
                walk(sub)
            return
        out.append(block)

    for b in blocks:
        walk(b)
    return out


def _resolve_inference_object_prim_path(
    task_cfg: dict[str, Any],
    override: str = "",
) -> str:
    """确定推理时要复位的物体 prim。

    优先级：
      1. ``--object-prim`` 命令行显式给定（**推荐**，与编排 YAML 完全解耦）
      2. 任务 YAML 里 ``skill_defaults.single_arm.pick.params.object_prim_path``
         （或 ``move_to_object``；scene_presets 可覆盖）

    为什么要支持 1：推理只需要「复位哪个物体」这一条信息，而任务 YAML 的
    ``task_queue`` 是给**编排执行**用的（20+ 步技能序列），推理一步也不执行。
    为了拿一个 prim 路径而要求任务里必须有完整编排队列，是概念上的耦合 ——
    用 ``--object-prim`` 就能把两者解耦。
    """
    if str(override or "").strip():
        return str(override).strip()

    from robot_action_composer.task_runtime.merge import merge_task_queue_skill_params  # pyright: ignore[reportMissingImports]

    task_queue = task_cfg.get("task_queue")
    if not isinstance(task_queue, list) or not task_queue:
        raise ValueError(
            "推理任务缺少 object_prim_path。两种解决办法：\n"
            "  1) 直接给 --object-prim <prim路径>（推荐，不需要编排 YAML）\n"
            "  2) 在任务 YAML 的 skill_defaults.single_arm.pick.params 里写 "
            "object_prim_path"
        )
    skill_defaults = dict(task_cfg.get("skill_defaults") or {})
    scene_presets = dict(task_cfg.get("scene_presets") or {})
    default_scene = task_cfg.get("default_scene")
    scene_sd = dict(scene_presets.get(default_scene) or {}) if isinstance(default_scene, str) else {}
    scene_skill_params = dict(scene_sd.get("skill_params") or {})

    merged_blocks = merge_task_queue_skill_params(list(task_queue), skill_defaults, scene_skill_params)
    for block in _iter_queue_blocks(merged_blocks):
        skill = str(block.get("skill") or "")
        if skill in {"single_arm.pick", "single_arm.move_to_object"}:
            params = dict(block.get("params") or {})
            raw = params.get("object_prim_path")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    raise ValueError(
        "Inference task missing object_prim_path for reset/source. "
        "Set it in skill_defaults.single_arm.pick (or move_to_object), "
        "then override per scene via scene_presets.<scene>.skill_params."
    )


def main() -> None:
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from robot_action_composer.discovery.registry_loader import (  # pyright: ignore[reportMissingImports]
        load_lerobot_profile,
        load_motion_entries,
        resolve_robot_profile_dir,
    )
    from robot_action_composer.task_config_io import (  # pyright: ignore[reportMissingImports]
        validate_runtime_defaults_keys,
    )

    from robot_action_composer.cli.interactive import (  # pyright: ignore[reportMissingImports]
        select_task_with_optional_group,
    )
    from online_infer.ui import select_option

    launcher_args, core_args = _parse_launcher_args()
    workspace_dir = (
        Path(launcher_args.workspace).expanduser().resolve()
        if launcher_args.workspace
        else isaac_dir
    )
    motion_registry = load_motion_entries(workspace_dir, robots_root=launcher_args.robots_root)
    if not motion_registry:
        raise RuntimeError(
            f"No motion/inference capable robot configs found from workspace {workspace_dir} "
            "(needs robots/, or a USD project with env/ + robot.yaml + task_configs/)"
        )

    robot_keys = list(motion_registry.keys())
    default_robot = launcher_args.robot or ("dobot_cr5" if "dobot_cr5" in motion_registry else robot_keys[0])
    robot_key = default_robot if launcher_args.robot else select_option(
        title="Select robot",
        options=motion_registry,
        default_key=default_robot,
    )
    if robot_key not in motion_registry:
        raise ValueError(f"Unknown robot key: {robot_key}")
    robot_entry = motion_registry[robot_key]

    task_options = {
        key: value
        for key, value in robot_entry.get("tasks", {}).items()
        if isinstance(value.get("runtime_defaults"), dict)
    }
    if not task_options:
        raise RuntimeError(f"No task configs found for robot: {robot_key}")

    task_keys = list(task_options.keys())
    default_task = launcher_args.task or ("pick_place" if "pick_place" in task_options else task_keys[0])
    task_key = (
        default_task
        if launcher_args.task
        else select_task_with_optional_group(
            title_group="Select task folder",
            title_task="Select inference task",
            tasks=task_options,
            task_groups=robot_entry.get("task_groups", {}),
            default_task_key=default_task,
        )
    )
    if task_key not in task_options:
        raise ValueError(f"Unknown task key: {task_key}")

    task_cfg = task_options[task_key]
    runtime_defaults = validate_runtime_defaults_keys(
        task_cfg.get("runtime_defaults", {}),
        context=f"task[{task_key}].runtime_defaults",
    )
    # 推理脚本**不再复位仿真**（开关/复位由人负责），所以不需要物体 prim 来等稳定。
    # 仍然解析它只为打印/兼容旧命令行参数。
    object_prim_path = str(launcher_args.object_prim or "").strip()
    if object_prim_path:
        print(f"[info] --object-prim={object_prim_path}（仅记录，不再触发复位）")

    core_script = isaac_dir / "online_infer" / "core.py"
    core_module = _load_module("isaac_inference_core", core_script)
    motion_cfg = robot_entry["motion_cfg"]
    lerobot_cfg = load_lerobot_profile(
        resolve_robot_profile_dir(workspace_dir, robot_entry),
        workspace_dir=workspace_dir,
    )
    if lerobot_cfg is None:
        raise RuntimeError(
            f"Robot {robot_key!r} has no lerobot_config.py; inference requires lerobot stack "
            "(./init.sh install-lerobot)."
        )
    bl_override = runtime_defaults.get("base_link_entity_path")
    if isinstance(bl_override, str) and bl_override.strip():
        if not is_dataclass(motion_cfg):
            raise TypeError(
                "runtime_defaults.base_link_entity_path requires motion_cfg to be a @dataclass "
                f"(got {type(motion_cfg).__name__})"
            )
        motion_cfg = replace(motion_cfg, base_link_entity_path=bl_override.strip())
    core_module.MOTION_CFG = motion_cfg
    core_module.LEROBOT_CFG = lerobot_cfg
    core_module.ROBOT_CFG = motion_cfg
    core_module.PICK_PLACE_FLOW_OVERRIDES = {
        "object_prim_path": object_prim_path,
        "no_reset": bool(getattr(launcher_args, "no_reset", False)),
    }

    print(
        f"[Selection] robot={robot_key}, task={task_key}, "
        f"source={object_prim_path}"
    )

    old_argv = sys.argv
    try:
        sys.argv = [str(core_script), *core_args]
        core_module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
