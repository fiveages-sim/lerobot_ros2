#!/usr/bin/env python3
"""Unified IsaacSim motion-generation launcher."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
import sys
from pathlib import Path
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


def _discover_flow_registry(isaac_dir: Path) -> dict[str, dict[str, Any]]:
    robots_root = isaac_dir / "robots"
    if not robots_root.is_dir():
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for robot_dir in sorted(p for p in robots_root.iterdir() if p.is_dir() and not p.name.startswith("__")):
        robot_cfg_file = robot_dir / "robot_config.py"
        flow_cfg_dir = robot_dir / "flow_configs"
        if not robot_cfg_file.is_file() or not flow_cfg_dir.is_dir():
            continue

        robot_mod = _load_module(f"{robot_dir.name}_robot_cfg", robot_cfg_file)
        robot_key = getattr(robot_mod, "ROBOT_KEY", robot_dir.name.lower())
        robot_label = getattr(robot_mod, "ROBOT_LABEL", robot_dir.name)
        robot_cfg = getattr(robot_mod, "ROBOT_CFG")

        tasks: dict[str, dict[str, Any]] = {}
        for flow_file in sorted(p for p in flow_cfg_dir.glob("*.py") if p.is_file()):
            flow_mod = _load_module(f"{robot_dir.name}_{flow_file.stem}_flow_cfg", flow_file)
            flow_cfg = getattr(flow_mod, "FLOW_CONFIG", None)
            if not isinstance(flow_cfg, dict):
                continue
            task_key = flow_cfg["task_key"]
            tasks[task_key] = dict(flow_cfg)

        if tasks:
            registry[robot_key] = {
                "label": robot_label,
                "robot_cfg": robot_cfg,
                "tasks": tasks,
            }

    return registry


def _select_option(*, title: str, options: list[str], default_value: str) -> str:
    print(f"\n{title}")
    for idx, name in enumerate(options, start=1):
        suffix = " (default)" if name == default_value else ""
        print(f"  {idx}. {name}{suffix}")
    raw = input("Select option (press Enter for default): ").strip()
    if raw == "":
        return default_value
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index]
    if raw in options:
        return raw
    print(f"[info] Invalid option '{raw}', using default '{default_value}'.")
    return default_value


def _apply_preset(base_cfg: Any, preset: dict[str, object]) -> Any:
    valid_fields = {f.name for f in fields(type(base_cfg))}
    unknown_keys = [k for k in preset if k not in valid_fields]
    if unknown_keys:
        raise ValueError(f"Unknown preset keys for {type(base_cfg).__name__}: {unknown_keys}")
    return replace(base_cfg, **preset)


def main() -> None:
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from handover_demo_common import (  # pyright: ignore[reportMissingImports]
        format_handover_task_cfg_summary,
        run_handover_demo,
    )
    from pick_place_flow_demo_common import (  # pyright: ignore[reportMissingImports]
        PickPlaceFlowTaskConfig,
        format_pick_place_flow_cfg_summary,
        run_pick_place_flow_demo,
    )
    from handover_demo_common import HandoverTaskConfig  # pyright: ignore[reportMissingImports]

    registry = _discover_flow_registry(isaac_dir)
    if not registry:
        raise RuntimeError("No motion-generation capable robot configs found under examples/IsaacSim/robots")

    print("IsaacSim Run Motion Generation")
    print("=" * 70)
    robot_keys = list(registry.keys())
    robot_key = _select_option(title="Select robot", options=robot_keys, default_value="dobot_cr5")
    robot_entry = registry[robot_key]

    task_keys = list(robot_entry["tasks"].keys())
    default_task = "pick_place_flow" if "pick_place_flow" in task_keys else task_keys[0]
    task_key = _select_option(title="Select task", options=task_keys, default_value=default_task)
    task_entry = robot_entry["tasks"][task_key]

    scene_presets: dict[str, dict[str, object]] = task_entry["scene_presets"]
    scene_names = list(scene_presets.keys())
    default_scene = task_entry["default_scene"]
    scene = _select_option(title="Select config", options=scene_names, default_value=default_scene)

    if task_entry["kind"] == "pick_place":
        base_task_cfg = PickPlaceFlowTaskConfig(**task_entry["base_task_overrides"])
        task_cfg = _apply_preset(base_task_cfg, scene_presets.get(scene, {}))
        print(format_pick_place_flow_cfg_summary(scene, task_cfg))
        run_pick_place_flow_demo(
            robot_cfg=robot_entry["robot_cfg"],
            task_cfg=task_cfg,
            robot_id=task_entry["robot_id"],
        )
        return

    if task_entry["kind"] == "handover":
        base_task_cfg = HandoverTaskConfig(**task_entry["base_task_overrides"])
        task_cfg = _apply_preset(base_task_cfg, scene_presets.get(scene, {}))
        print(format_handover_task_cfg_summary(scene, task_cfg))
        run_handover_demo(
            robot_cfg=robot_entry["robot_cfg"],
            handover_task_cfg=task_cfg,
            robot_id=task_entry["robot_id"],
        )
        return

    raise ValueError(f"Unsupported task kind: {task_entry['kind']}")


if __name__ == "__main__":
    main()
