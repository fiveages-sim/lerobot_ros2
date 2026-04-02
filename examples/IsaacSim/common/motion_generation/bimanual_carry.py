#!/usr/bin/env python3
"""Default IsaacSim bimanual carry demo flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from geometry_msgs.msg import Pose

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    StageTarget,
    build_bimanual_carry_sequence,
)

from demo_preset_utils import resolve_dataclass_cfg_from_presets


@dataclass(frozen=True)
class BimanualCarryTaskConfig:
    source_object_entity_path: str
    lateral_offset: float
    approach_offset: tuple[float, float, float]
    left_orientation: tuple[float, float, float, float]
    right_orientation: tuple[float, float, float, float]
    lift_offset: tuple[float, float, float]
    retreat_offset: tuple[float, float, float]
    lateral_clearance: float = 0.0
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_xyz_random_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def flatten_bimanual_carry_task_overrides(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Merge optional ``carry`` block into flat ``BimanualCarryTaskConfig`` kwargs.

    Top-level keys (except ``carry``) apply first; ``carry`` mapping overrides/extends them.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"carry overrides must be a mapping, got {type(raw).__name__}")
    merged: dict[str, Any] = {k: v for k, v in raw.items() if k != "carry"}
    carry = raw.get("carry")
    if carry is not None:
        if not isinstance(carry, Mapping):
            raise TypeError(f'"carry" must be a mapping, got {type(carry).__name__}')
        merged.update(dict(carry))
    return merged


def resolve_bimanual_carry_task_cfg_from_presets(
    *,
    base_task_cfg: BimanualCarryTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str = "default",
) -> tuple[str, BimanualCarryTaskConfig]:
    scene, task_cfg = resolve_dataclass_cfg_from_presets(
        base_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    return scene, task_cfg


def format_bimanual_carry_task_cfg_summary(
    scene: str, task_cfg: BimanualCarryTaskConfig,
) -> str:
    return (
        f"[Scene] {scene} -> {task_cfg.source_object_entity_path}, "
        f"lateral_offset={task_cfg.lateral_offset}, "
        f"approach_offset={task_cfg.approach_offset}, "
        f"lift_offset={task_cfg.lift_offset}, retreat_offset={task_cfg.retreat_offset}"
    )


def build_bimanual_carry_record_sequence(
    *,
    carry_task_cfg: BimanualCarryTaskConfig,
    object_center: Pose,
    gripper_open: float,
    gripper_closed: float,
) -> list[StageTarget]:
    """Build the full bimanual carry sequence as StageTarget list."""
    return build_bimanual_carry_sequence(
        object_center=object_center,
        lateral_offset=carry_task_cfg.lateral_offset,
        approach_offset=carry_task_cfg.approach_offset,
        lateral_clearance=carry_task_cfg.lateral_clearance,
        grasp_offset=carry_task_cfg.grasp_offset,
        left_orientation=carry_task_cfg.left_orientation,
        right_orientation=carry_task_cfg.right_orientation,
        lift_offset=carry_task_cfg.lift_offset,
        retreat_offset=carry_task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
    )
