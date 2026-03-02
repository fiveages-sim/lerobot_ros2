#!/usr/bin/env python3
"""Unified IsaacSim inference launcher."""

from __future__ import annotations

import argparse
import importlib.util
import sys
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

    from dataset_recording.launcher import select_option  # pyright: ignore[reportMissingImports]
    from robots.registry_loader import load_motion_entries  # pyright: ignore[reportMissingImports]

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
    task_key = default_task if launcher_args.task else select_option(
        title="Select inference task",
        options=task_options,
        default_key=default_task,
    )
    if task_key not in task_options:
        raise ValueError(f"Unknown task key: {task_key}")

    task_cfg = task_options[task_key]
    base_task_overrides = task_cfg.get("base_task_overrides", {})
    required_keys = {"source_object_entity_path", "object_xyz_random_offset"}
    if not required_keys.issubset(base_task_overrides.keys()):
        raise ValueError(
            f"Task '{task_key}' missing required fields for inference reset: {sorted(required_keys)}"
        )

    core_script = isaac_dir / "common" / "inference" / "core.py"
    core_module = _load_module("isaac_inference_core", core_script)
    core_module.ROBOT_CFG = robot_entry["robot_cfg"]
    core_module.PICK_PLACE_FLOW_OVERRIDES = base_task_overrides

    print(
        f"[Selection] robot={robot_key}, task={task_key}, "
        f"source={base_task_overrides['source_object_entity_path']}"
    )

    # Forward all remaining args to the legacy inference core parser.
    old_argv = sys.argv
    try:
        sys.argv = [str(core_script), *core_args]
        core_module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
