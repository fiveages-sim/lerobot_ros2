"""Unified motion generation builders and stage executors."""

from __future__ import annotations

import math
from typing import Any, Callable

from geometry_msgs.msg import Pose

ActionDict = dict[str, float]
StageAction = tuple[str, ActionDict]
DirectionVec = tuple[float, float, float]

PICK_STAGE_SUFFIXES: tuple[str, str, str, str] = ("Approach", "CloseIn", "Grasp", "Retreat")
PLACE_STAGE_SUFFIXES: tuple[str, str, str] = ("Place", "Release", "PostReleaseRetreat")
HANDOVER_STAGE_SUFFIXES: tuple[str, str, str] = ("SyncMove", "ReceiverGrasp", "SourceRelease")

_GRASP_DIRECTION_TO_VEC: dict[str, DirectionVec] = {
    "top": (0.0, 0.0, 1.0),
    "front": (-1.0, 0.0, 0.0),
    "back": (1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}


def _action_from_pose(
    pose: Pose,
    *,
    ee_prefix: str,
    gripper_key: str,
    gripper_value: float,
) -> ActionDict:
    return {
        f"{ee_prefix}.pos.x": pose.position.x,
        f"{ee_prefix}.pos.y": pose.position.y,
        f"{ee_prefix}.pos.z": pose.position.z,
        f"{ee_prefix}.quat.x": pose.orientation.x,
        f"{ee_prefix}.quat.y": pose.orientation.y,
        f"{ee_prefix}.quat.z": pose.orientation.z,
        f"{ee_prefix}.quat.w": pose.orientation.w,
        gripper_key: gripper_value,
    }


def _resolve_grasp_direction_vec(
    *,
    grasp_direction: str,
    grasp_direction_vector: DirectionVec | None,
) -> DirectionVec:
    if grasp_direction_vector is not None:
        vx, vy, vz = (float(grasp_direction_vector[0]), float(grasp_direction_vector[1]), float(grasp_direction_vector[2]))
    else:
        key = grasp_direction.lower()
        if key not in _GRASP_DIRECTION_TO_VEC:
            raise ValueError(
                f"Unsupported grasp_direction: {grasp_direction}. "
                f"Expected one of: {', '.join(sorted(_GRASP_DIRECTION_TO_VEC))}"
            )
        vx, vy, vz = _GRASP_DIRECTION_TO_VEC[key]

    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm < 1e-8:
        raise ValueError("grasp_direction_vector norm is too small")
    return (vx / norm, vy / norm, vz / norm)


def _make_pose_from_target(
    target_pose: Pose,
    *,
    offset: float,
    direction_vec: DirectionVec,
    orientation: tuple[float, float, float, float],
) -> Pose:
    pose = Pose()
    dx, dy, dz = direction_vec
    pose.position.x = target_pose.position.x + dx * offset
    pose.position.y = target_pose.position.y + dy * offset
    pose.position.z = target_pose.position.z + dz * offset
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
    return pose


def _make_pose(
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
    return pose


def _merge_actions(*actions: ActionDict) -> ActionDict:
    out: ActionDict = {}
    for action in actions:
        out.update(action)
    return out


def _stage_name(prefix: str, index: int, suffix: str) -> str:
    return f"{prefix}-{index}-{suffix}"


def build_single_arm_pick_sequence(
    *,
    target_pose: Pose,
    approach_clearance: float,
    grasp_clearance: float,
    grasp_orientation: tuple[float, float, float, float],
    grasp_direction: str = "top",
    grasp_direction_vector: DirectionVec | None = None,
    retreat_direction_extra: float = 0.0,
    retreat_raise_z: float = 0.0,
    gripper_open: float,
    gripper_closed: float,
    ee_prefix: str,
    gripper_key: str,
    stage_prefix: str = "Pickup",
) -> list[StageAction]:
    direction_vec = _resolve_grasp_direction_vec(
        grasp_direction=grasp_direction,
        grasp_direction_vector=grasp_direction_vector,
    )
    approach_pose = _make_pose_from_target(
        target_pose,
        offset=approach_clearance,
        direction_vec=direction_vec,
        orientation=grasp_orientation,
    )
    close_in_pose = _make_pose_from_target(
        target_pose,
        offset=grasp_clearance,
        direction_vec=direction_vec,
        orientation=grasp_orientation,
    )
    retreat_pose = _make_pose_from_target(
        target_pose,
        offset=approach_clearance + retreat_direction_extra,
        direction_vec=direction_vec,
        orientation=grasp_orientation,
    )
    retreat_pose.position.z += retreat_raise_z

    return [
        (
            _stage_name(stage_prefix, 1, PICK_STAGE_SUFFIXES[0]),
            _action_from_pose(
                approach_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
        (
            _stage_name(stage_prefix, 2, PICK_STAGE_SUFFIXES[1]),
            _action_from_pose(
                close_in_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
        (
            _stage_name(stage_prefix, 3, PICK_STAGE_SUFFIXES[2]),
            _action_from_pose(
                close_in_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_closed,
            ),
        ),
        (
            _stage_name(stage_prefix, 4, PICK_STAGE_SUFFIXES[3]),
            _action_from_pose(
                retreat_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_closed,
            ),
        ),
    ]


def build_single_arm_place_sequence(
    *,
    place_position: tuple[float, float, float],
    place_orientation: tuple[float, float, float, float],
    post_release_retract_offset: tuple[float, float, float],
    gripper_open: float,
    gripper_closed: float,
    ee_prefix: str,
    gripper_key: str,
    stage_prefix: str = "Place",
    start_index: int = 1,
) -> list[StageAction]:
    place_pose = _make_pose(place_position, place_orientation)
    retract_pose = _make_pose(
        (
            place_position[0] + post_release_retract_offset[0],
            place_position[1] + post_release_retract_offset[1],
            place_position[2] + post_release_retract_offset[2],
        ),
        place_orientation,
    )
    return [
        (
            _stage_name(stage_prefix, start_index, PLACE_STAGE_SUFFIXES[0]),
            _action_from_pose(
                place_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_closed,
            ),
        ),
        (
            _stage_name(stage_prefix, start_index + 1, PLACE_STAGE_SUFFIXES[1]),
            _action_from_pose(
                place_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
        (
            _stage_name(stage_prefix, start_index + 2, PLACE_STAGE_SUFFIXES[2]),
            _action_from_pose(
                retract_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
    ]


def build_single_arm_return_home_sequence(
    *,
    home_action: ActionDict,
    stage_name: str = "Return-1-ReturnHome",
) -> list[StageAction]:
    return [(stage_name, dict(home_action))]


def build_handover_sequence_for_arms(
    *,
    source_handover_pose: Pose,
    receiver_handover_pose: Pose,
    source_ee_prefix: str,
    source_gripper_key: str,
    receiver_ee_prefix: str,
    receiver_gripper_key: str,
    gripper_open: float,
    gripper_closed: float,
    stage_prefix: str = "Handover",
) -> list[StageAction]:
    receiver_handover_open = _action_from_pose(
        receiver_handover_pose,
        ee_prefix=receiver_ee_prefix,
        gripper_key=receiver_gripper_key,
        gripper_value=gripper_open,
    )
    receiver_handover_closed = _action_from_pose(
        receiver_handover_pose,
        ee_prefix=receiver_ee_prefix,
        gripper_key=receiver_gripper_key,
        gripper_value=gripper_closed,
    )
    source_handover_closed = _action_from_pose(
        source_handover_pose,
        ee_prefix=source_ee_prefix,
        gripper_key=source_gripper_key,
        gripper_value=gripper_closed,
    )
    source_handover_open = _action_from_pose(
        source_handover_pose,
        ee_prefix=source_ee_prefix,
        gripper_key=source_gripper_key,
        gripper_value=gripper_open,
    )
    return [
        (
            _stage_name(stage_prefix, 1, HANDOVER_STAGE_SUFFIXES[0]),
            _merge_actions(receiver_handover_open, source_handover_closed),
        ),
        (
            _stage_name(stage_prefix, 2, HANDOVER_STAGE_SUFFIXES[1]),
            _merge_actions(receiver_handover_closed, source_handover_closed),
        ),
        (
            _stage_name(stage_prefix, 3, HANDOVER_STAGE_SUFFIXES[2]),
            _merge_actions(receiver_handover_closed, source_handover_open),
        ),
    ]


def compose_bimanual_synchronized_sequence(
    left_sequence: list[StageAction],
    right_sequence: list[StageAction],
) -> list[StageAction]:
    if len(left_sequence) != len(right_sequence):
        raise ValueError(
            "Left/right sequences must have the same number of stages for synchronized composition"
        )

    merged: list[StageAction] = []
    for idx, ((left_name, left_action), (right_name, right_action)) in enumerate(
        zip(left_sequence, right_sequence),
        start=1,
    ):
        stage_label = f"{idx:02d}-{left_name}|{right_name}"
        action = dict(left_action)
        for key, value in right_action.items():
            if key in action:
                raise ValueError(f"Conflicting action key during bimanual merge: {key}")
            action[key] = value
        merged.append((stage_label, action))
    return merged


def execute_stage_sequence(
    *,
    robot: Any,
    sequence: list[StageAction],
    wait_both_arms: bool,
    arrival_timeout: float,
    arrival_poll: float,
    time_now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    gripper_action_wait: float,
    single_arm_part: str = "right_arm",
    left_arrival_guard_stage: str | None = None,
    warn_prefix: str = "Stage timeout",
) -> None:
    for stage_name, action in sequence:
        print(f"[Stage] {stage_name}")
        robot.send_action(action)
        if wait_both_arms:
            left_arrive = robot.ros2_interface.wait_until_arrive(
                part="left_arm",
                timeout=arrival_timeout,
                poll_period=arrival_poll,
                time_now_fn=time_now_fn,
                sleep_fn=sleep_fn,
            )
            right_arrive = robot.ros2_interface.wait_until_arrive(
                part="right_arm",
                timeout=arrival_timeout,
                poll_period=arrival_poll,
                time_now_fn=time_now_fn,
                sleep_fn=sleep_fn,
            )
        else:
            single_arrive = robot.ros2_interface.wait_until_arrive(
                part=single_arm_part,
                timeout=arrival_timeout,
                poll_period=arrival_poll,
                time_now_fn=time_now_fn,
                sleep_fn=sleep_fn,
            )
            if single_arm_part == "left_arm":
                left_arrive = single_arrive
                right_arrive = {"arrived": True}
            else:
                right_arrive = single_arrive
                left_arrive = {"arrived": True}

        if "Grasp" in stage_name or "Release" in stage_name:
            sleep_fn(gripper_action_wait)
        if not left_arrive.get("arrived", False) or not right_arrive.get("arrived", False):
            print(
                f"[WARN] {warn_prefix}: "
                f"left_arrived={left_arrive.get('arrived', False)}, "
                f"right_arrived={right_arrive.get('arrived', False)}"
            )
        if left_arrival_guard_stage and stage_name == left_arrival_guard_stage and not left_arrive.get(
            "arrived", False
        ):
            raise RuntimeError("Left arm did not arrive at guarded stage; skip subsequent grasp.")


__all__ = [
    "build_single_arm_pick_sequence",
    "build_single_arm_place_sequence",
    "build_single_arm_return_home_sequence",
    "build_handover_sequence_for_arms",
    "compose_bimanual_synchronized_sequence",
    "execute_stage_sequence",
]
