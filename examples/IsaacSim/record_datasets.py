#!/usr/bin/env python3
"""Unified IsaacSim dataset recording entrypoint."""

from __future__ import annotations

from dataclasses import fields, replace
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
            flatten_pick_place_task_overrides,
        )

        flat = flatten_pick_place_task_overrides(task_entry_cfg["base_task_overrides"])
        return PickPlaceFlowTaskConfig(**flat)
    if task_kind == "handover":
        from motion_generation.handover import (  # pyright: ignore[reportMissingImports]
            HandoverTaskConfig,
            flatten_handover_task_overrides,
        )

        flat = flatten_handover_task_overrides(task_entry_cfg["base_task_overrides"])
        return HandoverTaskConfig(**flat)
    if task_kind == "bimanual_carry":
        from motion_generation.bimanual_carry import (  # pyright: ignore[reportMissingImports]
            BimanualCarryTaskConfig,
            flatten_bimanual_carry_task_overrides,
        )

        flat = flatten_bimanual_carry_task_overrides(task_entry_cfg["base_task_overrides"])
        return BimanualCarryTaskConfig(**flat)
    return task_entry_cfg


def _apply_task_preset(base_cfg: Any, preset: dict[str, object]) -> Any:
    if not preset:
        return base_cfg
    if not hasattr(type(base_cfg), "__dataclass_fields__"):
        if isinstance(base_cfg, dict):
            merged = dict(base_cfg)
            merged.update(preset)
            return merged
        return base_cfg
    valid_fields = {f.name for f in fields(type(base_cfg))}
    unknown_keys = [k for k in preset if k not in valid_fields]
    if unknown_keys:
        raise ValueError(f"Unknown preset keys for {type(base_cfg).__name__}: {unknown_keys}")
    return replace(base_cfg, **preset)


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
                "use_stamped": task_entry.get("use_stamped", True),
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
    scene_presets: dict[str, dict[str, object]] = task_entry["task_cfg"].get("scene_presets", {})
    default_scene = task_entry["task_cfg"].get("default_scene")
    if scene_presets:
        scene_keys = list(scene_presets.keys())
        if default_scene not in scene_presets:
            default_scene = scene_keys[0]
        scene_options = {scene: {"label": scene} for scene in scene_keys}
        scene_key = select_option(
            title="Select scene",
            options=scene_options,
            default_key=default_scene,
        )
        preset_raw: dict[str, object] = scene_presets.get(scene_key, {})
        if task_entry.get("task_kind") == "pick_place":
            from motion_generation.pick_place import (  # pyright: ignore[reportMissingImports]
                flatten_pick_place_task_overrides,
            )

            preset_raw = flatten_pick_place_task_overrides(preset_raw)
        elif task_entry.get("task_kind") == "handover":
            from motion_generation.handover import (  # pyright: ignore[reportMissingImports]
                flatten_handover_task_overrides,
            )

            preset_raw = flatten_handover_task_overrides(preset_raw)
        elif task_entry.get("task_kind") == "bimanual_carry":
            from motion_generation.bimanual_carry import (  # pyright: ignore[reportMissingImports]
                flatten_bimanual_carry_task_overrides,
            )

            preset_raw = flatten_bimanual_carry_task_overrides(preset_raw)
        task_cfg = _apply_task_preset(task_cfg, preset_raw)
    else:
        scene_key = "default"

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
        f"[Selection] robot={robot_key}, task={task_key}, scene={scene_key}, profile={profile_key}, "
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
        use_stamped=bool(task_entry.get("use_stamped", True)),
    )


if __name__ == "__main__":
    main()
