#!/usr/bin/env python3
"""Default IsaacSim bimanual handover demo flow."""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

from ros2_robot_interface import FSM_HOLD, FSM_OCS2  # pyright: ignore[reportMissingImports]

from lerobot_robot_ros2 import (
    ROS2Robot,
    ROS2RobotConfig,
    build_handover_sequence_for_arms,
    build_single_arm_pick_sequence,
    build_single_arm_place_sequence,
    build_single_arm_return_home_sequence,
    execute_stage_sequence,
)
from lerobot_robot_ros2.utils import action_from_pose, obs_to_pose, pose_from_tuple  # pyright: ignore[reportMissingImports]

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
        "handover_demo only supports gripper_control_mode='target_command' without explicit gripper values"
    )


def _merge_actions(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    out.update(b)
    return out


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
    source_target_pose: Any,
    source_home_pose: Any,
    receiver_home_pose: Any,
    source_is_right: bool,
    gripper_open: float,
    gripper_closed: float,
) -> list[tuple[str, dict[str, float]]]:
    source_ee_prefix = "right_ee" if source_is_right else "left_ee"
    source_gripper_key = "right_gripper.pos" if source_is_right else "left_gripper.pos"
    receiver_ee_prefix = "left_ee" if source_is_right else "right_ee"
    receiver_gripper_key = "left_gripper.pos" if source_is_right else "right_gripper.pos"

    sequence: list[tuple[str, dict[str, float]]] = []
    sequence.extend(
        build_single_arm_pick_sequence(
            target_pose=source_target_pose,
            approach_clearance=handover_task_cfg.approach_clearance,
            grasp_clearance=handover_task_cfg.grasp_clearance,
            grasp_orientation=handover_task_cfg.grasp_orientation,
            grasp_direction=handover_task_cfg.grasp_direction,
            grasp_direction_vector=handover_task_cfg.grasp_direction_vector,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            ee_prefix=source_ee_prefix,
            gripper_key=source_gripper_key,
            stage_prefix="Pickup",
        )
    )

    source_handover_pose = pose_from_tuple(
        handover_task_cfg.handover_position,
        handover_task_cfg.source_handover_orientation,
    )
    receiver_handover_pose = pose_from_tuple(
        handover_task_cfg.handover_position,
        handover_task_cfg.receiver_handover_orientation,
    )
    receiver_handover_pose.position.z -= 0.02
    receiver_handover_pose.position.y -= 0.02
    handover_primitive = build_handover_sequence_for_arms(
        source_handover_pose=source_handover_pose,
        receiver_handover_pose=receiver_handover_pose,
        source_ee_prefix=source_ee_prefix,
        source_gripper_key=source_gripper_key,
        receiver_ee_prefix=receiver_ee_prefix,
        receiver_gripper_key=receiver_gripper_key,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        stage_prefix="Handover",
    )
    sequence.extend(handover_primitive[:3])

    if handover_task_cfg.run_place_after_handover:
        place_primitive = build_single_arm_place_sequence(
            place_position=handover_task_cfg.receiver_place_position,
            place_orientation=handover_task_cfg.receiver_place_orientation,
            post_release_retract_offset=(0.0, 0.0, 0.0),
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
            ee_prefix=receiver_ee_prefix,
            gripper_key=receiver_gripper_key,
            stage_prefix="Place",
            start_index=1,
        )
        source_home_open = action_from_pose(
            source_home_pose,
            gripper_open,
            ee_prefix=source_ee_prefix,
            gripper_key=source_gripper_key,
        )
        receiver_return_home = build_single_arm_return_home_sequence(
            home_action=action_from_pose(
                receiver_home_pose,
                gripper_open,
                ee_prefix=receiver_ee_prefix,
                gripper_key=receiver_gripper_key,
            ),
            stage_name="Place-4-ReceiverReturnHome",
        )
        sequence.extend(
            [
                (place_primitive[0][0], _merge_actions(place_primitive[0][1], source_home_open)),
                (place_primitive[1][0], _merge_actions(place_primitive[1][1], source_home_open)),
                (place_primitive[2][0], _merge_actions(place_primitive[2][1], source_home_open)),
                ("Place-4-ReceiverReturnHome", _merge_actions(receiver_return_home[0][1], source_home_open)),
            ]
        )
    return sequence


def run_handover_demo(
    *,
    robot_cfg: Any,
    handover_task_cfg: HandoverTaskConfig,
    robot_id: str = "isaac_bimanual_handover",
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
        initial_arm = handover_task_cfg.initial_grasp_arm.lower()
        if initial_arm not in {"left", "right"}:
            raise ValueError("initial_grasp_arm must be 'left' or 'right'")
        source_is_right = initial_arm == "right"
        source_ee_prefix = "right_ee" if source_is_right else "left_ee"
        source_gripper_key = "right_gripper.pos" if source_is_right else "left_gripper.pos"
        receiver_ee_prefix = "left_ee" if source_is_right else "right_ee"
        receiver_gripper_key = "left_gripper.pos" if source_is_right else "right_gripper.pos"
        gripper_open, gripper_closed = _resolve_gripper_open_close(
            robot_cfg.gripper_control_mode
        )

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

        pregrasp_action = action_from_pose(
            source_home_pose,
            gripper_open,
            ee_prefix=source_ee_prefix,
            gripper_key=source_gripper_key,
        )
        robot.send_action(pregrasp_action)
        sim_time.sleep(robot_cfg.gripper_action_wait)

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
        sequence = [item for item in full_sequence if item[0].startswith("Pickup-")]
        execute_stage_sequence(
            robot=robot,
            sequence=sequence,
            wait_both_arms=False,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            time_now_fn=sim_time.now_seconds,
            sleep_fn=sim_time.sleep,
            gripper_action_wait=robot_cfg.gripper_action_wait,
            single_arm_part="right_arm" if source_is_right else "left_arm",
            warn_prefix="Stage timeout",
        )

        handover_sequence = [item for item in full_sequence if item[0].startswith("Handover-")]
        execute_stage_sequence(
            robot=robot,
            sequence=handover_sequence,
            wait_both_arms=True,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            time_now_fn=sim_time.now_seconds,
            sleep_fn=sim_time.sleep,
            gripper_action_wait=robot_cfg.gripper_action_wait,
            left_arrival_guard_stage="Handover-1-SyncMove" if source_is_right else None,
            warn_prefix="Handover stage timeout",
        )
        place_sequence = [item for item in full_sequence if item[0].startswith("Place-")]
        if place_sequence:
            execute_stage_sequence(
                robot=robot,
                sequence=place_sequence,
                wait_both_arms=True,
                arrival_timeout=robot_cfg.arrival_timeout,
                arrival_poll=robot_cfg.arrival_poll,
                time_now_fn=sim_time.now_seconds,
                sleep_fn=sim_time.sleep,
                gripper_action_wait=robot_cfg.gripper_action_wait,
                warn_prefix="Place stage timeout",
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        print("[OK] Bimanual grasp demo completed")
    except Exception as err:
        print(f"[ERROR] Demo failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")

