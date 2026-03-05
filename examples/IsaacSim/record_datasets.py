#!/usr/bin/env python3
"""Unified IsaacSim dataset recording entrypoint."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from typing import Any


def _build_task_config(task_entry_cfg: Any) -> Any:
    if not (isinstance(task_entry_cfg, dict) and "base_task_overrides" in task_entry_cfg):
        return task_entry_cfg
    task_kind = task_entry_cfg.get("kind")
    if task_kind == "pick_place":
        from motion_generation.pick_place import (  # pyright: ignore[reportMissingImports]
            PickPlaceFlowTaskConfig,
        )

        return PickPlaceFlowTaskConfig(**task_entry_cfg["base_task_overrides"])
    if task_kind == "handover":
        from motion_generation.handover import HandoverTaskConfig  # pyright: ignore[reportMissingImports]

        return HandoverTaskConfig(**task_entry_cfg["base_task_overrides"])
    return task_entry_cfg


def _pointcloud_supported(robot_cfg: Any) -> bool:
    cameras = getattr(robot_cfg, "cameras", {}) or {}
    has_depth_topic = any(getattr(cam, "depth_topic_name", None) for cam in cameras.values())
    has_depth_info = getattr(robot_cfg, "depth_info_topic", None) is not None
    return bool(has_depth_topic and has_depth_info)

def main() -> None:
    # Extend module search paths.
    isaac_dir = Path(__file__).resolve().parent
    common_dir = isaac_dir / "common"
    for path in (isaac_dir, common_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from dataset_recording.launcher import (
        collect_runtime_options,
        select_option,
    )
    from robots.registry_loader import load_robot_entries

    discovered = load_robot_entries(isaac_dir)
    registry: dict[str, dict[str, Any]] = {}
    for key, entry in discovered.items():
        record = entry["record"]
        if not record:
            continue
        tasks: dict[str, dict[str, Any]] = {}
        for task_key, task_entry in record["tasks"].items():
            base_cfg = task_entry["record_cfg"]
            profiles = {}
            for profile in task_entry.get("profiles", []):
                overrides = profile.get("overrides", {})
                cfg = replace(base_cfg, **overrides) if overrides else base_cfg
                profiles[profile["key"]] = {
                    "label": profile.get("label", profile["key"]),
                    "record_cfg": cfg,
                }
            tasks[task_key] = {
                "label": task_entry.get("label", task_key),
                "task_cfg": task_entry["task_cfg"],
                "task_kind": task_entry.get("task_kind", task_entry.get("task_cfg", {}).get("kind")),
                "runner": task_entry["runner"],
                "record_profiles": profiles,
            }
        registry[key] = {
            "label": entry["label"],
            "robot_cfg": entry["robot_cfg"],
            "tasks": tasks,
        }
    if not registry:
        raise RuntimeError("No record-capable robot configs found under examples/IsaacSim/robots")

    print("IsaacSim Record Datasets")
    print("=" * 70)
    robot_keys = list(registry.keys())
    default_robot = "dobot_cr5" if "dobot_cr5" in registry else robot_keys[0]
    robot_key = select_option(title="Select robot", options=registry, default_key=default_robot)
    robot_entry = registry[robot_key]

    task_keys = list(robot_entry["tasks"].keys())
    default_task = "pick_place" if "pick_place" in robot_entry["tasks"] else task_keys[0]
    task_key = select_option(
        title="Select task",
        options=robot_entry["tasks"],
        default_key=default_task,
    )
    task_entry = robot_entry["tasks"][task_key]
    task_cfg = _build_task_config(task_entry["task_cfg"])

    profile_keys = list(task_entry["record_profiles"].keys())
    default_profile = "default" if "default" in task_entry["record_profiles"] else profile_keys[0]
    profile_key = select_option(
        title="Select record profile",
        options=task_entry["record_profiles"],
        default_key=default_profile,
    )
    profile_entry = task_entry["record_profiles"][profile_key]

    loops, enable_keypoint_pcd, enable_manual_episode_check = collect_runtime_options(
        pointcloud_supported=_pointcloud_supported(robot_entry["robot_cfg"]),
        default_enable_keypoint_pcd=False,
    )
    print(
        f"[Selection] robot={robot_key}, task={task_key}, profile={profile_key}, "
        f"episodes={loops}, pointcloud={enable_keypoint_pcd}, manual_review={enable_manual_episode_check}"
    )
    task_entry["runner"](
        robot_cfg=robot_entry["robot_cfg"],
        task_cfg=task_cfg,
        record_cfg=profile_entry["record_cfg"],
        loops=loops,
        enable_keypoint_pcd=enable_keypoint_pcd,
        enable_manual_episode_check=enable_manual_episode_check,
        task_kind=task_entry["task_kind"],
        task_name=task_key,
    )


if __name__ == "__main__":
    main()
