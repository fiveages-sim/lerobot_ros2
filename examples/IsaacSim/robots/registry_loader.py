#!/usr/bin/env python3
"""Auto-discovery loader for IsaacSim robot/task/record configs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util
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


def _ensure_package_paths(isaac_dir: Path) -> None:
    repo_root = isaac_dir.parents[1]
    for pkg_root in (repo_root / "lerobot_camera_ros2", repo_root / "lerobot_robot_ros2"):
        if pkg_root.is_dir() and str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))


def load_motion_entries(isaac_dir: Path) -> dict[str, dict[str, Any]]:
    _ensure_package_paths(isaac_dir)
    robots_root = isaac_dir / "robots"
    entries: dict[str, dict[str, Any]] = {}
    if not robots_root.is_dir():
        return entries

    for robot_dir in sorted(p for p in robots_root.iterdir() if p.is_dir() and not p.name.startswith("__")):
        robot_cfg_file = robot_dir / "robot_config.py"
        task_cfg_dir = robot_dir / "task_configs"
        if not robot_cfg_file.is_file() or not task_cfg_dir.is_dir():
            continue

        robot_mod = _load_module(f"{robot_dir.name}_robot_cfg", robot_cfg_file)
        robot_key = getattr(robot_mod, "ROBOT_KEY", robot_dir.name.lower())
        robot_label = getattr(robot_mod, "ROBOT_LABEL", robot_dir.name)
        robot_cfg = getattr(robot_mod, "ROBOT_CFG")

        tasks: dict[str, dict[str, Any]] = {}
        for task_file in sorted(p for p in task_cfg_dir.glob("*.py") if p.is_file()):
            task_mod = _load_module(f"{robot_dir.name}_{task_file.stem}_task_cfg", task_file)
            task_cfg = getattr(task_mod, "TASK_CONFIG", None)
            if task_cfg is None:
                task_cfg = getattr(task_mod, "FLOW_CONFIG", None)
            if not isinstance(task_cfg, dict):
                continue
            task_key = task_cfg["task_key"]
            tasks[task_key] = dict(task_cfg)

        entries[robot_key] = {
            "label": robot_label,
            "robot_cfg": robot_cfg,
            "tasks": tasks,
        }
    return entries


def load_robot_entries(isaac_dir: Path) -> dict[str, dict[str, Any]]:
    _ensure_package_paths(isaac_dir)

    motion_entries = load_motion_entries(isaac_dir)
    common_runner_mod = _load_module(
        "isaac_record_runner",
        isaac_dir / "common" / "dataset_recording" / "runner.py",
    )
    runner = getattr(common_runner_mod, "run_recording")
    supported_kinds = set(getattr(common_runner_mod, "SUPPORTED_RECORD_KINDS", ()))
    base_record_cfg = getattr(common_runner_mod, "DEFAULT_RECORD_CFG", None)
    if base_record_cfg is None:
        raise RuntimeError("dataset_recording.runner must define DEFAULT_RECORD_CFG")
    entries: dict[str, dict[str, Any]] = {}
    if not motion_entries:
        return entries

    for robot_key, motion_entry in motion_entries.items():
        robot_label = motion_entry["label"]
        robot_cfg = motion_entry["robot_cfg"]
        flow_tasks: dict[str, dict[str, Any]] = motion_entry.get("tasks", {})
        record_entry = None
        record_tasks: dict[str, dict[str, Any]] = {}
        for task_key, flow_cfg in flow_tasks.items():
            task_kind = flow_cfg.get("kind")
            if supported_kinds and task_kind not in supported_kinds:
                continue
            record_cfg_section = flow_cfg.get("record", {})
            if not record_cfg_section:
                continue
            base_overrides = record_cfg_section.get("base_record_overrides", {})
            task_record_cfg = replace(base_record_cfg, **base_overrides) if base_overrides else base_record_cfg
            profiles = record_cfg_section.get(
                "profiles",
                [{"key": "default", "label": "Default", "overrides": {}}],
            )
            record_tasks[task_key] = {
                "label": flow_cfg.get("label", task_key),
                "task_cfg": flow_cfg,
                "task_kind": task_kind,
                "runner": runner,
                "record_cfg": task_record_cfg,
                "profiles": profiles,
            }
        if record_tasks:
            record_entry = {"tasks": record_tasks}

        entries[robot_key] = {
            "label": robot_label,
            "robot_cfg": robot_cfg,
            "record": record_entry,
        }
    return entries
