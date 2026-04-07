#!/usr/bin/env python3
"""Default IsaacSim bimanual handover demo flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from geometry_msgs.msg import Pose

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    ArmSide,
    ArmStage,
    ArmTarget,
    StageTarget,
    assign_to_arm,
    build_handover_sequence,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
)

from demo_preset_utils import resolve_dataclass_cfg_from_presets


def _pose_from_tuple(
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
    return pose


@dataclass(frozen=True)
class HandoverTaskConfig:
    initial_grasp_arm: str
    grasp_orientation: tuple[float, float, float, float]
    object_xyz_random_offset: tuple[float, float, float]
    approach_clearance: float
    grasp_clearance: float
    source_object_entity_path: str
    handover_position: tuple[float, float, float]
    source_handover_orientation: tuple[float, float, float, float]
    receiver_handover_orientation: tuple[float, float, float, float]
    receiver_place_position: tuple[float, float, float]
    receiver_place_orientation: tuple[float, float, float, float]
    grasp_direction: str = "top"
    grasp_direction_vector: tuple[float, float, float] | None = None
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    retreat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    receiver_handover_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    run_place_after_handover: bool = True


def flatten_handover_task_overrides(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Merge optional ``pick`` / ``handover`` / ``place`` blocks into flat ``HandoverTaskConfig`` kwargs.

    Precedence for duplicate keys: top-level (excluding section keys) < ``pick`` < ``handover`` <
    ``place`` (later wins).
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"handover overrides must be a mapping, got {type(raw).__name__}")
    merged: dict[str, Any] = {
        k: v for k, v in raw.items() if k not in ("pick", "handover", "place")
    }
    for section in ("pick", "handover", "place"):
        block = raw.get(section)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise TypeError(f'"{section}" must be a mapping, got {type(block).__name__}')
        merged.update(dict(block))
    return merged


def resolve_handover_task_cfg_from_presets(
    *,
    base_task_cfg: HandoverTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str = "grab_medicine",
) -> tuple[str, HandoverTaskConfig]:
    scene, task_cfg = resolve_dataclass_cfg_from_presets(
        base_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    return scene, task_cfg


def format_handover_task_cfg_summary(scene: str, task_cfg: HandoverTaskConfig) -> str:
    return (
        f"[Scene] {scene} -> {task_cfg.source_object_entity_path}, "
        f"arm={task_cfg.initial_grasp_arm}, direction={task_cfg.grasp_direction}, "
        f"orientation={task_cfg.grasp_orientation}, "
        f"approach_clearance={task_cfg.approach_clearance}, grasp_clearance={task_cfg.grasp_clearance}, "
        f"grasp_offset={task_cfg.grasp_offset}, retreat_offset={task_cfg.retreat_offset}, "
        f"object_xyz_random_offset={task_cfg.object_xyz_random_offset}"
    )


def build_handover_record_sequence(
    *,
    handover_task_cfg: HandoverTaskConfig,
    source_target_pose: Pose,
    source_home_pose: Pose,
    receiver_home_pose: Pose,
    source_is_right: bool,
    gripper_open: float,
    gripper_closed: float,
) -> list[StageTarget]:
    """Build the full pick → handover → place sequence as StageTarget list."""
    source_arm = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
    receiver_arm = ArmSide.LEFT if source_is_right else ArmSide.RIGHT

    # --- Pickup (single-arm) ---
    pick_arm_stages = build_single_arm_pick_sequence(
        target_pose=source_target_pose,
        approach_clearance=handover_task_cfg.approach_clearance,
        grasp_clearance=handover_task_cfg.grasp_clearance,
        grasp_orientation=handover_task_cfg.grasp_orientation,
        grasp_direction=handover_task_cfg.grasp_direction,
        grasp_direction_vector=handover_task_cfg.grasp_direction_vector,
        grasp_offset=handover_task_cfg.grasp_offset,
        retreat_offset=handover_task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="Pickup",
    )
    sequence: list[StageTarget] = assign_to_arm(pick_arm_stages, source_arm)

    # --- Handover (bimanual) ---
    source_handover_pose = _pose_from_tuple(
        handover_task_cfg.handover_position,
        handover_task_cfg.source_handover_orientation,
    )
    receiver_handover_pose = _pose_from_tuple(
        handover_task_cfg.handover_position,
        handover_task_cfg.receiver_handover_orientation,
    )
    rx, ry, rz = handover_task_cfg.receiver_handover_offset
    receiver_handover_pose.position.x += rx
    receiver_handover_pose.position.y += ry
    receiver_handover_pose.position.z += rz
    handover_stages = build_handover_sequence(
        source_handover_pose=source_handover_pose,
        receiver_handover_pose=receiver_handover_pose,
        source_arm=source_arm,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="Handover",
    )
    sequence.extend(handover_stages)

    # --- Place (bimanual: receiver places, source returns home) ---
    if handover_task_cfg.run_place_after_handover:
        place_arm_stages = build_single_arm_place_sequence(
            place_position=handover_task_cfg.receiver_place_position,
            place_orientation=handover_task_cfg.receiver_place_orientation,
            post_release_retract_offset=(0.0, 0.0, 0.0),
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            stage_prefix="Place",
            start_index=1,
        )
        receiver_return_home = build_single_arm_return_home_sequence(
            home_pose=receiver_home_pose,
            gripper=gripper_open,
            stage_name="Place-4-ReceiverReturnHome",
        )
        source_home_target = ArmTarget(pose=source_home_pose, gripper=gripper_open)
        all_place_arm_stages: list[ArmStage] = list(place_arm_stages) + list(receiver_return_home)
        for stage_name, receiver_target in all_place_arm_stages:
            # Keep a single reference frame per stage:
            # - regular place stages use base-link targets (receiver only),
            # - return-home stage can use ee-frame home poses for both arms.
            if "ReceiverReturnHome" in stage_name:
                if source_arm == ArmSide.LEFT:
                    sequence.append(StageTarget(name=stage_name, left=source_home_target, right=receiver_target))
                else:
                    sequence.append(StageTarget(name=stage_name, left=receiver_target, right=source_home_target))
            else:
                if receiver_arm == ArmSide.LEFT:
                    sequence.append(StageTarget(name=stage_name, left=receiver_target))
                else:
                    sequence.append(StageTarget(name=stage_name, right=receiver_target))

    return sequence
