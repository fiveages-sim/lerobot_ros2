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
    return parser.parse_known_args()


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


def _resolve_inference_object_prim_path(task_cfg: dict[str, Any]) -> str:
    from robot_action_composer.task_runtime.merge import merge_task_queue_skill_params  # pyright: ignore[reportMissingImports]

    task_queue = task_cfg.get("task_queue")
    if not isinstance(task_queue, list) or not task_queue:
        raise ValueError("Inference task requires a non-empty task_queue.")
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

    from robot_action_composer.discovery.registry_loader import load_motion_entries  # pyright: ignore[reportMissingImports]
    from robot_action_composer.task_config_io import (  # pyright: ignore[reportMissingImports]
        validate_runtime_defaults_keys,
    )

    from robot_action_composer.dataset_recording.launcher import (  # pyright: ignore[reportMissingImports]
        select_task_with_optional_group,
    )
    from online_infer.ui import select_option

    launcher_args, core_args = _parse_launcher_args()
    motion_registry = load_motion_entries(isaac_dir)
    if not motion_registry:
        raise RuntimeError("No motion/inference capable robot configs found under examples/IsaacSim/robots")

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
    object_prim_path = _resolve_inference_object_prim_path(task_cfg)

    core_script = isaac_dir / "online_infer" / "core.py"
    core_module = _load_module("isaac_inference_core", core_script)
    robot_cfg = robot_entry["robot_cfg"]
    bl_override = runtime_defaults.get("base_link_entity_path")
    if isinstance(bl_override, str) and bl_override.strip():
        if not is_dataclass(robot_cfg):
            raise TypeError(
                "runtime_defaults.base_link_entity_path requires robot_cfg to be a @dataclass "
                f"(got {type(robot_cfg).__name__})"
            )
        robot_cfg = replace(robot_cfg, base_link_entity_path=bl_override.strip())
    core_module.ROBOT_CFG = robot_cfg
    core_module.PICK_PLACE_FLOW_OVERRIDES = {"object_prim_path": object_prim_path}

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
