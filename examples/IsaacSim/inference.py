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


def main() -> None:
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from robot_action_composer.discovery.registry_loader import load_motion_entries  # pyright: ignore[reportMissingImports]
    from robot_action_composer.task_config_io import flatten_queue_task_overrides  # pyright: ignore[reportMissingImports]
    from robot_action_composer.task_runtime.merge import (  # pyright: ignore[reportMissingImports]
        merge_flat_with_skill_carry,
        merge_flat_with_skill_drawer,
        merge_flat_with_skill_handover,
        merge_flat_with_skill_pick_place,
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
        if isinstance(value.get("base_task_overrides"), dict)
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
    base_task_overrides = merge_flat_with_skill_pick_place(
        flatten_queue_task_overrides(task_cfg.get("base_task_overrides", {})),
        task_cfg.get("skill_defaults"),
    )
    base_task_overrides = merge_flat_with_skill_handover(
        base_task_overrides,
        task_cfg.get("skill_defaults"),
    )
    base_task_overrides = merge_flat_with_skill_carry(
        base_task_overrides,
        task_cfg.get("skill_defaults"),
    )
    base_task_overrides = merge_flat_with_skill_drawer(
        base_task_overrides,
        task_cfg.get("skill_defaults"),
    )
    required_keys = {"source_object_entity_path", "object_xyz_random_offset"}
    if not required_keys.issubset(base_task_overrides.keys()):
        raise ValueError(
            f"Task '{task_key}' missing required fields for inference reset: {sorted(required_keys)}"
        )

    core_script = isaac_dir / "online_infer" / "core.py"
    core_module = _load_module("isaac_inference_core", core_script)
    robot_cfg = robot_entry["robot_cfg"]
    bl_override = base_task_overrides.get("base_link_entity_path")
    if isinstance(bl_override, str) and bl_override.strip():
        if not is_dataclass(robot_cfg):
            raise TypeError(
                "base_task_overrides.base_link_entity_path requires robot_cfg to be a @dataclass "
                f"(got {type(robot_cfg).__name__})"
            )
        robot_cfg = replace(robot_cfg, base_link_entity_path=bl_override.strip())
    core_module.ROBOT_CFG = robot_cfg
    core_module.PICK_PLACE_FLOW_OVERRIDES = base_task_overrides

    print(
        f"[Selection] robot={robot_key}, task={task_key}, "
        f"source={base_task_overrides['source_object_entity_path']}"
    )

    old_argv = sys.argv
    try:
        sys.argv = [str(core_script), *core_args]
        core_module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
