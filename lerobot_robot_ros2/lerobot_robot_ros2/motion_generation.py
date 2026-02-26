"""Backend-agnostic motion generation helpers for single/dual arm tasks."""

from __future__ import annotations

from geometry_msgs.msg import Pose

from .utils.pose_utils import action_from_pose


ActionDict = dict[str, float]
StageAction = tuple[str, ActionDict]


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


def build_single_arm_grasp_transport_release_sequence(
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
    ee_prefix: str = "left_ee",
    gripper_key: str = "left_gripper.pos",
    home_action: ActionDict | None = None,
    home_stage_name: str = "9-ReturnHome",
) -> list[StageAction]:
    """Build a staged single-arm grasp/transport/release sequence."""
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

    sequence: list[StageAction] = [
        (
            "1-Approach",
            action_from_pose(
                approach_pose,
                gripper_open,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "2-Descend",
            action_from_pose(
                descend_pose,
                gripper_open,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "3-Grasp",
            action_from_pose(
                descend_pose,
                gripper_closed,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "4-Lift",
            action_from_pose(
                lift_pose,
                gripper_closed,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "5-Transport",
            action_from_pose(
                transport_pose,
                gripper_closed,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "6-Lower",
            action_from_pose(
                lower_pose,
                gripper_closed,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "7-Release",
            action_from_pose(
                lower_pose,
                gripper_open,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
        (
            "8-Retract",
            action_from_pose(
                retract_pose,
                gripper_open,
                ee_prefix=ee_prefix,
                gripper_key=gripper_key,
            ),
        ),
    ]
    if home_action is not None:
        sequence.append((home_stage_name, dict(home_action)))
    return sequence


def compose_bimanual_synchronized_sequence(
    left_sequence: list[StageAction],
    right_sequence: list[StageAction],
) -> list[StageAction]:
    """Synchronize two per-arm staged sequences into shared bimanual stages.

    This keeps planner scope minimal: it only handles stage-level orchestration.
    Low-level interpolation/timing remains in the control stack.
    """
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
