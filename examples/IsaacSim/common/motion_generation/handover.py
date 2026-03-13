#!/usr/bin/env python3
"""Default IsaacSim bimanual handover demo flow."""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

from geometry_msgs.msg import Pose

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    FSM_HOLD,
    FSM_OCS2,
    ArmSide,
    ArmStage,
    ArmTarget,
    SendMode,
    StageTarget,
    assign_to_arm,
    build_handover_sequence,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
    execute_stage_sequence,
)

from lerobot_robot_ros2 import (
    ROS2Robot,
    ROS2RobotConfig,
)
from lerobot_robot_ros2.utils import obs_to_pose  # pyright: ignore[reportMissingImports]

from isaac_ros2_sim_common import (
    SimTimeHelper,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
)
from demo_preset_utils import resolve_dataclass_cfg_from_presets
from motion_generation.movej_return import (
    capture_initial_arm_joint_positions,
    movej_return_to_initial_state,
)


def _resolve_gripper_open_close(mode: str) -> tuple[float, float]:
    if mode == "target_command":
        return 1.0, 0.0
    raise ValueError(
        "handover_demo only supports gripper_control_mode='target_command' without explicit gripper values"
    )


def _make_robot(robot_cfg: Any, robot_id: str) -> ROS2Robot:
    cfg = getattr(robot_cfg, "ros2_interface", None)
    if cfg is None:
        raise ValueError("robot_cfg must define ros2_interface (ROS2RobotInterfaceConfig)")
    return ROS2Robot(
        ROS2RobotConfig(
            id=robot_id,
            ros2_interface=cfg,
            gripper_control_mode=robot_cfg.gripper_control_mode,
        )
    )


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
    receiver_handover_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    run_place_after_handover: bool = True


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
            if source_arm == ArmSide.LEFT:
                sequence.append(StageTarget(name=stage_name, left=source_home_target, right=receiver_target))
            else:
                sequence.append(StageTarget(name=stage_name, left=receiver_target, right=source_home_target))

    return sequence


def run_handover_demo(
    *,
    robot_cfg: Any,
    handover_task_cfg: HandoverTaskConfig,
    robot_id: str = "isaac_bimanual_handover",
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Run default IsaacSim handover flow with robot-specific config."""
    robot = _make_robot(robot_cfg, robot_id)
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        try:
            if robot_connected:
                robot.disconnect()
        finally:
            sim_time.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        robot.connect()
        robot_connected = True
        print("[OK] Robot connected")
        left_initial_joint_positions, right_initial_joint_positions = capture_initial_arm_joint_positions(robot)
        initial_arm = handover_task_cfg.initial_grasp_arm.lower()
        if initial_arm not in {"left", "right"}:
            raise ValueError("initial_grasp_arm must be 'left' or 'right'")
        source_is_right = initial_arm == "right"
        source_arm = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
        gripper_open, gripper_closed = _resolve_gripper_open_close(
            robot_cfg.gripper_control_mode
        )
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        if reset_env:
            reset_simulation_and_randomize_object(
                handover_task_cfg.source_object_entity_path,
                xyz_offset=handover_task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        left_home_pose = obs_to_pose(obs0, "left_ee")
        right_home_pose = obs_to_pose(obs0, "right_ee")
        source_home_pose = right_home_pose if source_is_right else left_home_pose
        receiver_home_pose = left_home_pose if source_is_right else right_home_pose
        source_ee_prefix = "right_ee" if source_is_right else "left_ee"

        # Pre-grasp: open source gripper at home position.
        # Use the frame_id from the ee pose topic (observation frame).
        source_handler = (
            robot.ros2_interface.right_arm_handler if source_is_right
            else robot.ros2_interface.left_arm_handler
        )
        ee_frame_id = (source_handler.frame_id if source_handler else None) or frame_id
        pregrasp_stage = assign_to_arm(
            [("pregrasp", ArmTarget(pose=source_home_pose, gripper=gripper_open))],
            source_arm,
        )
        execute_stage_sequence(
            interface=robot.ros2_interface,
            sequence=pregrasp_stage,
            send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
            frame_id=ee_frame_id,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            time_now_fn=sim_time.now_seconds,
            sleep_fn=sim_time.sleep,
            gripper_action_wait=robot_cfg.gripper_action_wait,
        )

        base_world_pos, base_world_quat = get_entity_pose_world_service(
            robot_cfg.base_link_entity_path
        )
        source_target_pose = get_object_pose_from_service(
            base_world_pos,
            base_world_quat,
            handover_task_cfg.source_object_entity_path,
            include_orientation=False,
        )

        full_sequence = build_handover_record_sequence(
            handover_task_cfg=handover_task_cfg,
            source_target_pose=source_target_pose,
            source_home_pose=source_home_pose,
            receiver_home_pose=receiver_home_pose,
            source_is_right=source_is_right,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
        )

        # Split: pickup is single-arm (stamped), handover+place are dual-arm (dual stamped)
        pickup_stages = [s for s in full_sequence if s.name.startswith("Pickup-")]
        bimanual_stages = [s for s in full_sequence if not s.name.startswith("Pickup-")]

        if use_stamped:
            for stage in bimanual_stages:
                if "ReturnHome" in stage.name:
                    stage.frame_id = ee_frame_id

        common_kw = dict(
            interface=robot.ros2_interface,
            frame_id=frame_id,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            time_now_fn=sim_time.now_seconds,
            sleep_fn=sim_time.sleep,
            gripper_action_wait=robot_cfg.gripper_action_wait,
        )

        if pickup_stages:
            execute_stage_sequence(
                **common_kw,
                sequence=pickup_stages,
                send_mode=SendMode.STAMPED if use_stamped else SendMode.UNSTAMPED,
                warn_prefix="Pickup stage timeout",
            )

        if bimanual_stages:
            execute_stage_sequence(
                **common_kw,
                sequence=bimanual_stages,
                send_mode=SendMode.DUAL_ARM_STAMPED if use_stamped else SendMode.UNSTAMPED,
                left_arrival_guard_stage="Handover-1-SyncMove" if source_is_right else None,
                warn_prefix="Handover stage timeout",
            )

        moved_by_movej = movej_return_to_initial_state(
            robot=robot,
            left_initial_positions=left_initial_joint_positions,
            right_initial_positions=right_initial_joint_positions,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            sim_time=sim_time,
        )
        if not moved_by_movej:
            print("[WARN] MoveJ return-to-initial skipped: no valid initial arm joints.")
        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Bimanual grasp demo completed")
    except Exception as err:
        print(f"[ERROR] Demo failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")
