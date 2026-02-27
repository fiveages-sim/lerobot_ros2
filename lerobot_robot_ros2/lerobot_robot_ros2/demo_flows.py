"""Reusable demo flow builders and stage executors."""

from __future__ import annotations

import math
from typing import Any, Callable

from geometry_msgs.msg import Pose

ActionDict = dict[str, float]
StageAction = tuple[str, ActionDict]
DirectionVec = tuple[float, float, float]

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


def build_single_arm_pickup_sequence(
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
) -> list[StageAction]:
    """Build a staged single-arm pickup sequence (approach/close-in/grasp/retreat)."""
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
    descend_pose = _make_pose_from_target(
        target_pose,
        offset=grasp_clearance,
        direction_vec=direction_vec,
        orientation=grasp_orientation,
    )
    lift_pose = _make_pose_from_target(
        target_pose,
        offset=approach_clearance,
        direction_vec=direction_vec,
        orientation=grasp_orientation,
    )
    if retreat_direction_extra != 0.0 or retreat_raise_z != 0.0:
        retreat_pose = _make_pose_from_target(
            target_pose,
            offset=approach_clearance + retreat_direction_extra,
            direction_vec=direction_vec,
            orientation=grasp_orientation,
        )
        retreat_pose.position.z += retreat_raise_z
    else:
        retreat_pose = lift_pose

    return [
        (
            "Pickup-1-Approach",
            _action_from_pose(
                approach_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
        (
            "Pickup-2-CloseIn",
            _action_from_pose(
                descend_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_open,
            ),
        ),
        (
            "Pickup-3-Grasp",
            _action_from_pose(
                descend_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_closed,
            ),
        ),
        (
            "Pickup-4-Retreat",
            _action_from_pose(
                retreat_pose,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
                gripper_value=gripper_closed,
            ),
        ),
    ]


def build_handover_then_place_sequence_for_arms(
    *,
    source_handover_pose: Pose,
    receiver_handover_pose: Pose,
    receiver_place_pose: Pose,
    source_home_pose: Pose,
    receiver_home_pose: Pose,
    source_ee_prefix: str,
    source_gripper_key: str,
    receiver_ee_prefix: str,
    receiver_gripper_key: str,
    gripper_open: float,
    gripper_closed: float,
) -> list[StageAction]:
    """Build handover + place sequence for arbitrary source/receiver arms."""
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
    receiver_place_closed = _action_from_pose(
        receiver_place_pose,
        ee_prefix=receiver_ee_prefix,
        gripper_key=receiver_gripper_key,
        gripper_value=gripper_closed,
    )
    receiver_place_open = _action_from_pose(
        receiver_place_pose,
        ee_prefix=receiver_ee_prefix,
        gripper_key=receiver_gripper_key,
        gripper_value=gripper_open,
    )
    receiver_home_open = _action_from_pose(
        receiver_home_pose,
        ee_prefix=receiver_ee_prefix,
        gripper_key=receiver_gripper_key,
        gripper_value=gripper_open,
    )
    source_home_open = _action_from_pose(
        source_home_pose,
        ee_prefix=source_ee_prefix,
        gripper_key=source_gripper_key,
        gripper_value=gripper_open,
    )

    def merge(a: ActionDict, b: ActionDict) -> ActionDict:
        out = dict(a)
        out.update(b)
        return out

    return [
        ("Handover-1-SyncMove", merge(receiver_handover_open, source_handover_closed)),
        ("Handover-3-ReceiverGrasp", merge(receiver_handover_closed, source_handover_closed)),
        ("Handover-4-SourceRelease", merge(receiver_handover_closed, source_handover_open)),
        ("Handover-5-ReceiverPlaceAndSourceHome", merge(receiver_place_closed, source_home_open)),
        ("Handover-6-ReceiverReleaseAtPlace", merge(receiver_place_open, source_home_open)),
        ("Handover-7-ReceiverReturnHome", merge(receiver_home_open, source_home_open)),
    ]


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
    """Execute staged actions with arrival checks and optional guard."""
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
