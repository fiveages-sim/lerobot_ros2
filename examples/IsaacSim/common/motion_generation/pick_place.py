#!/usr/bin/env python3
"""Generic IsaacSim pick-place flow common implementation."""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    FSM_HOLD,
    FSM_OCS2,
    ArmSide,
    ArmStage,
    ArmTarget,
    StageTarget,
    assign_to_arm,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
    compose_bimanual_synchronized_sequence,
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


def _resolve_gripper_open_close(mode: str) -> tuple[float, float]:
    if mode == "target_command":
        return 1.0, 0.0
    raise ValueError(
        "pick_place_flow only supports gripper_control_mode='target_command' without explicit gripper values"
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


@dataclass(frozen=True)
class PickPlaceFlowTaskConfig:
    mode: str = "single_arm"  # single_arm | dual_arm
    initial_grasp_arm: str = "left"
    object_xyz_random_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_pose_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_clearance: float = 0.2
    grasp_clearance: float = 0.01
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    retreat_direction_extra: float = 0.0
    retreat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_object_entity_path: str = ""
    left_object_entity_path: str | None = None
    right_object_entity_path: str | None = None
    grasp_orientation: tuple[float, float, float, float] = (-0.7, 0.7, 0.0, 0.0)
    grasp_direction: str = "top"
    grasp_direction_vector: tuple[float, float, float] | None = None
    left_grasp_orientation: tuple[float, float, float, float] | None = None
    right_grasp_orientation: tuple[float, float, float, float] | None = None
    left_grasp_direction: str | None = None
    right_grasp_direction: str | None = None
    left_grasp_direction_vector: tuple[float, float, float] | None = None
    right_grasp_direction_vector: tuple[float, float, float] | None = None
    run_place_before_return: bool = False
    place_position: tuple[float, float, float] | None = None
    place_orientation: tuple[float, float, float, float] | None = None
    post_release_retract_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_stage_duration: float = 2.0
    pose_tol_pos: float = 0.025
    pose_tol_ori: float = 0.08
    require_orientation_reach: bool = False
    use_object_orientation: bool = False


def resolve_pick_place_cfg_from_presets(
    *,
    base_task_cfg: PickPlaceFlowTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str = "grab_medicine",
) -> tuple[str, PickPlaceFlowTaskConfig]:
    scene, task_cfg = resolve_dataclass_cfg_from_presets(
        base_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    return scene, task_cfg


def format_pick_place_cfg_summary(scene: str, task_cfg: PickPlaceFlowTaskConfig) -> str:
    return (
        f"[Scene] {scene} -> {task_cfg.source_object_entity_path}, "
        f"arm={task_cfg.initial_grasp_arm}, direction={task_cfg.grasp_direction}, "
        f"orientation={task_cfg.grasp_orientation}, "
        f"approach_clearance={task_cfg.approach_clearance}, grasp_clearance={task_cfg.grasp_clearance}, "
        f"target_pose_offset={task_cfg.target_pose_offset}, "
        f"retreat_direction_extra={task_cfg.retreat_direction_extra}, retreat_offset={task_cfg.retreat_offset}"
    )


def _apply_target_pose_offset(pose: Any, offset: tuple[float, float, float]) -> Any:
    ox, oy, oz = offset
    pose.position.x += ox
    pose.position.y += oy
    pose.position.z += oz
    return pose


def _resolve_arm_grasp_config(
    task_cfg: PickPlaceFlowTaskConfig,
    *,
    is_right: bool,
) -> tuple[tuple[float, float, float, float], str, tuple[float, float, float] | None]:
    if is_right:
        orientation = task_cfg.right_grasp_orientation or task_cfg.grasp_orientation
        direction = task_cfg.right_grasp_direction or task_cfg.grasp_direction
        direction_vector = (
            task_cfg.right_grasp_direction_vector
            if task_cfg.right_grasp_direction_vector is not None
            else task_cfg.grasp_direction_vector
        )
    else:
        orientation = task_cfg.left_grasp_orientation or task_cfg.grasp_orientation
        direction = task_cfg.left_grasp_direction or task_cfg.grasp_direction
        direction_vector = (
            task_cfg.left_grasp_direction_vector
            if task_cfg.left_grasp_direction_vector is not None
            else task_cfg.grasp_direction_vector
        )
    return orientation, direction, direction_vector


def build_single_arm_pick_place_sequence(
    *,
    target_pose: Any,
    home_pose: Any,
    task_cfg: PickPlaceFlowTaskConfig,
    arm_side: ArmSide,
    gripper_open: float,
    gripper_closed: float,
    grasp_orientation: tuple[float, float, float, float],
    grasp_direction: str,
    grasp_direction_vector: tuple[float, float, float] | None,
) -> list[StageTarget]:
    """Build a single-arm pick (+ optional place) + return-home sequence."""
    arm_seq: list[ArmStage] = list(build_single_arm_pick_sequence(
        target_pose=target_pose,
        approach_clearance=task_cfg.approach_clearance,
        grasp_clearance=task_cfg.grasp_clearance,
        grasp_orientation=grasp_orientation,
        grasp_direction=grasp_direction,
        grasp_direction_vector=grasp_direction_vector,
        grasp_offset=task_cfg.grasp_offset,
        retreat_direction_extra=task_cfg.retreat_direction_extra,
        retreat_offset=task_cfg.retreat_offset,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="PickPlaceFlow",
    ))
    if task_cfg.run_place_before_return:
        if task_cfg.place_position is None or task_cfg.place_orientation is None:
            raise ValueError("place_position and place_orientation are required when run_place_before_return=True")
        arm_seq.extend(
            build_single_arm_place_sequence(
                place_position=task_cfg.place_position,
                place_orientation=task_cfg.place_orientation,
                post_release_retract_offset=task_cfg.post_release_retract_offset,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                stage_prefix="PickPlaceFlow",
                start_index=5,
            )
        )
        return_stage_name = "PickPlaceFlow-8-ReturnHomeHold"
    else:
        return_stage_name = "PickPlaceFlow-5-ReturnHomeHold"
    arm_seq.extend(
        build_single_arm_return_home_sequence(
            home_pose=home_pose,
            gripper=gripper_closed,
            stage_name=return_stage_name,
        )
    )
    return assign_to_arm(arm_seq, arm_side)


def run_pick_place_demo(
    *,
    robot_cfg: Any,
    task_cfg: PickPlaceFlowTaskConfig,
    robot_id: str = "isaac_pick_place_flow",
) -> None:
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
        mode = task_cfg.mode.lower()
        if mode not in {"single_arm", "dual_arm"}:
            raise ValueError("mode must be 'single_arm' or 'dual_arm'")
        if mode == "dual_arm" and task_cfg.run_place_before_return:
            raise ValueError("run_place_before_return is currently only supported in single_arm mode")

        gripper_open, gripper_closed = _resolve_gripper_open_close(robot_cfg.gripper_control_mode)

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        base_world_pos, base_world_quat = get_entity_pose_world_service(robot_cfg.base_link_entity_path)

        if mode == "single_arm":
            initial_arm = task_cfg.initial_grasp_arm.lower()
            if initial_arm not in {"left", "right"}:
                raise ValueError("initial_grasp_arm must be 'left' or 'right'")
            source_is_right = initial_arm == "right"
            arm_side = ArmSide.RIGHT if source_is_right else ArmSide.LEFT
            source_ee_prefix = "right_ee" if source_is_right else "left_ee"
            source_home_pose = obs_to_pose(obs0, source_ee_prefix)
            source_object_path = task_cfg.source_object_entity_path
            if not source_object_path:
                raise ValueError("source_object_entity_path is required for single_arm mode")

            reset_simulation_and_randomize_object(
                source_object_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

            # Pre-grasp: open gripper at home position
            pregrasp_target = ArmTarget(pose=source_home_pose, gripper=gripper_open)
            pregrasp_stage = assign_to_arm(
                [("pregrasp", pregrasp_target)], arm_side,
            )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=pregrasp_stage,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
            )

            source_target_pose = get_object_pose_from_service(
                base_world_pos,
                base_world_quat,
                source_object_path,
                include_orientation=False,
            )
            source_target_pose = _apply_target_pose_offset(source_target_pose, task_cfg.target_pose_offset)
            source_grasp_orientation, source_grasp_direction, source_grasp_direction_vector = (
                _resolve_arm_grasp_config(task_cfg, is_right=source_is_right)
            )
            sequence = build_single_arm_pick_place_sequence(
                target_pose=source_target_pose,
                home_pose=source_home_pose,
                task_cfg=task_cfg,
                arm_side=arm_side,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=source_grasp_orientation,
                grasp_direction=source_grasp_direction,
                grasp_direction_vector=source_grasp_direction_vector,
            )
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=sequence,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )
        else:
            left_home_pose = obs_to_pose(obs0, "left_ee")
            right_home_pose = obs_to_pose(obs0, "right_ee")
            left_object_path = task_cfg.left_object_entity_path
            right_object_path = task_cfg.right_object_entity_path
            if not left_object_path or not right_object_path:
                raise ValueError("left_object_entity_path and right_object_entity_path are required for dual_arm mode")

            reset_simulation_and_randomize_object(
                left_object_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )
            reset_simulation_and_randomize_object(
                right_object_path,
                xyz_offset=task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

            # Pre-grasp: open both grippers at home positions
            pregrasp_stage = [StageTarget(
                name="pregrasp",
                left=ArmTarget(pose=left_home_pose, gripper=gripper_open),
                right=ArmTarget(pose=right_home_pose, gripper=gripper_open),
            )]
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=pregrasp_stage,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
            )

            left_target_pose = get_object_pose_from_service(
                base_world_pos, base_world_quat, left_object_path, include_orientation=False
            )
            right_target_pose = get_object_pose_from_service(
                base_world_pos, base_world_quat, right_object_path, include_orientation=False
            )
            left_target_pose = _apply_target_pose_offset(left_target_pose, task_cfg.target_pose_offset)
            right_target_pose = _apply_target_pose_offset(right_target_pose, task_cfg.target_pose_offset)
            left_grasp_orientation, left_grasp_direction, left_grasp_direction_vector = (
                _resolve_arm_grasp_config(task_cfg, is_right=False)
            )
            right_grasp_orientation, right_grasp_direction, right_grasp_direction_vector = (
                _resolve_arm_grasp_config(task_cfg, is_right=True)
            )
            left_seq = build_single_arm_pick_place_sequence(
                target_pose=left_target_pose,
                home_pose=left_home_pose,
                task_cfg=task_cfg,
                arm_side=ArmSide.LEFT,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=left_grasp_orientation,
                grasp_direction=left_grasp_direction,
                grasp_direction_vector=left_grasp_direction_vector,
            )
            right_seq = build_single_arm_pick_place_sequence(
                target_pose=right_target_pose,
                home_pose=right_home_pose,
                task_cfg=task_cfg,
                arm_side=ArmSide.RIGHT,
                gripper_open=gripper_open,
                gripper_closed=gripper_closed,
                grasp_orientation=right_grasp_orientation,
                grasp_direction=right_grasp_direction,
                grasp_direction_vector=right_grasp_direction_vector,
            )
            # Extract ArmStage lists for bimanual composition
            left_arm_stages: list[ArmStage] = [(s.name, s.left) for s in left_seq if s.left]  # type: ignore[misc]
            right_arm_stages: list[ArmStage] = [(s.name, s.right) for s in right_seq if s.right]  # type: ignore[misc]
            merged_seq = compose_bimanual_synchronized_sequence(left_arm_stages, right_arm_stages)
            execute_stage_sequence(
                interface=robot.ros2_interface,
                sequence=merged_seq,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="PickPlaceFlow timeout",
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] PickPlaceFlow demo completed")
    except Exception as err:
        print(f"[ERROR] PickPlaceFlow failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


def run_pick_place_entry(
    *,
    robot_cfg: Any,
    base_task_cfg: PickPlaceFlowTaskConfig,
    scene_presets: dict[str, dict[str, object]],
    cli_description: str,
    default_scene: str,
    robot_id: str,
) -> None:
    """Thin entry wrapper for robot-specific pick-place-flow scripts."""
    scene, task_cfg = resolve_pick_place_cfg_from_presets(
        base_task_cfg=base_task_cfg,
        scene_presets=scene_presets,
        cli_description=cli_description,
        default_scene=default_scene,
    )
    print(format_pick_place_cfg_summary(scene, task_cfg))
    run_pick_place_demo(
        robot_cfg=robot_cfg,
        task_cfg=task_cfg,
        robot_id=robot_id,
    )
