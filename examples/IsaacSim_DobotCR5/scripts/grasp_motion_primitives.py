#!/usr/bin/env python3
"""Reusable grasp/transport/release motion primitives."""

from __future__ import annotations

from geometry_msgs.msg import Pose
from lerobot_robot_ros2.utils.pose_utils import action_from_pose  # pyright: ignore[reportMissingImports]


def _make_pose(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def build_grasp_transport_release_sequence(
    target_pose: Pose,
    *,
    approach_clearance: float,
    grasp_clearance: float,
    grasp_orientation: tuple[float, float, float, float],
    place_orientation: tuple[float, float, float, float],
    release_position: tuple[float, float, float],
    transport_height: float,
    retract_offset_y: float,
    gripper_open: float,
    gripper_closed: float,
    home_action: dict[str, float] | None = None,
    home_stage_name: str = "9-ReturnHome",
) -> list[tuple[str, dict[str, float]]]:
    """Build grasp/transport/release sequence, optionally including return-home."""
    gqx, gqy, gqz, gqw = grasp_orientation
    pqx, pqy, pqz, pqw = place_orientation
    rx, ry, rz = release_position

    approach_pose = _make_pose(
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z + approach_clearance,
        gqx,
        gqy,
        gqz,
        gqw,
    )
    descend_pose = _make_pose(
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z + grasp_clearance,
        gqx,
        gqy,
        gqz,
        gqw,
    )
    lift_pose = _make_pose(
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z + approach_clearance,
        gqx,
        gqy,
        gqz,
        gqw,
    )
    transport_pose = _make_pose(
        rx,
        ry,
        transport_height,
        pqx,
        pqy,
        pqz,
        pqw,
    )
    lower_pose = _make_pose(
        rx,
        ry,
        rz,
        pqx,
        pqy,
        pqz,
        pqw,
    )
    retract_pose = _make_pose(
        rx,
        ry + retract_offset_y,
        rz,
        pqx,
        pqy,
        pqz,
        pqw,
    )

    sequence = [
        ("1-Approach", action_from_pose(approach_pose, gripper_open)),
        ("2-Descend", action_from_pose(descend_pose, gripper_open)),
        ("3-Grasp", action_from_pose(descend_pose, gripper_closed)),
        ("4-Lift", action_from_pose(lift_pose, gripper_closed)),
        ("5-Transport", action_from_pose(transport_pose, gripper_closed)),
        ("6-Lower", action_from_pose(lower_pose, gripper_closed)),
        ("7-Release", action_from_pose(lower_pose, gripper_open)),
        ("8-Retract", action_from_pose(retract_pose, gripper_open)),
    ]
    if home_action is not None:
        sequence.append((home_stage_name, home_action))
    return sequence
