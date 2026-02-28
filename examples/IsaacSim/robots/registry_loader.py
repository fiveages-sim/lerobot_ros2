#!/usr/bin/env python3
"""Auto-discovery loader for IsaacSim robot manifests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys
from typing import Any


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclass internals may consult sys.modules during class creation.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_attr(*, script_dir: Path, module_file: str, attr: str, unique_name: str) -> Any:
    module = _load_module(unique_name, script_dir / module_file)
    if not hasattr(module, attr):
        raise AttributeError(f"Module '{module_file}' has no attr '{attr}'")
    return getattr(module, attr)


def load_robot_entries(isaac_dir: Path) -> dict[str, dict[str, Any]]:
    robots_dir = isaac_dir / "robots"
    manifest_files = sorted(
        p
        for p in robots_dir.glob("*.json")
        if p.is_file()
    )
    entries: dict[str, dict[str, Any]] = {}
    for manifest_file in manifest_files:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        robot_key = data["robot_key"]
        script_dir = isaac_dir / data["script_dir"]
        robot_cfg = _load_attr(
            script_dir=script_dir,
            module_file=data["robot_cfg"]["module_file"],
            attr=data["robot_cfg"]["attr"],
            unique_name=f"{robot_key}_robot_cfg",
        )

        flow_tasks: dict[str, dict[str, Any]] = {}
        for task in data.get("flow_tasks", []):
            base_task_cfg = _load_attr(
                script_dir=script_dir,
                module_file=task["module_file"],
                attr=task["base_task_cfg_attr"],
                unique_name=f"{robot_key}_{task['task_key']}_flow",
            )
            if task.get("scene_presets_attr"):
                scene_presets = _load_attr(
                    script_dir=script_dir,
                    module_file=task["module_file"],
                    attr=task["scene_presets_attr"],
                    unique_name=f"{robot_key}_{task['task_key']}_scenes",
                )
            else:
                scene_presets = {"default": {}}

            flow_tasks[task["task_key"]] = {
                "label": task.get("label", task["task_key"]),
                "kind": task["kind"],
                "base_task_cfg": base_task_cfg,
                "scene_presets": scene_presets,
                "default_scene": task["default_scene"],
                "robot_id": task["robot_id"],
            }

        record_entry = None
        if data.get("record"):
            record_cfg = _load_attr(
                script_dir=script_dir,
                module_file=data["record"]["record_cfg_module_file"],
                attr=data["record"]["record_cfg_attr"],
                unique_name=f"{robot_key}_record_cfg",
            )
            runner = _load_attr(
                script_dir=script_dir,
                module_file=data["record"]["runner_module_file"],
                attr=data["record"]["runner_attr"],
                unique_name=f"{robot_key}_record_runner",
            )
            record_tasks: dict[str, dict[str, Any]] = {}
            for task in data["record"].get("tasks", []):
                task_cfg = _load_attr(
                    script_dir=script_dir,
                    module_file=task["module_file"],
                    attr=task["task_cfg_attr"],
                    unique_name=f"{robot_key}_{task['task_key']}_record_task",
                )
                record_tasks[task["task_key"]] = {
                    "label": task.get("label", task["task_key"]),
                    "task_cfg": task_cfg,
                    "runner": runner,
                }
            record_entry = {
                "record_cfg": record_cfg,
                "profiles": data["record"].get("profiles", []),
                "tasks": record_tasks,
            }

        entries[robot_key] = {
            "label": data["label"],
            "robot_cfg": robot_cfg,
            "flow_tasks": flow_tasks,
            "record": record_entry,
        }
    return entries
