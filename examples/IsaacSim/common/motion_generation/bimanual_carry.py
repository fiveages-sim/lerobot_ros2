#!/usr/bin/env python3
"""Default IsaacSim bimanual carry demo flow."""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

from geometry_msgs.msg import Pose

from ros2_robot_interface import (  # pyright: ignore[reportMissingImports]
    FSM_HOLD,
    FSM_OCS2,
    ArmTarget,
    SendMode,
    StageTarget,
    build_bimanual_carry_sequence,
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
        "bimanual_carry_demo only supports gripper_control_mode='target_command' without explicit gripper values"
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


def run_bimanual_carry_demo(
    *,
    robot_cfg: Any,
    carry_task_cfg: BimanualCarryTaskConfig,
    robot_id: str = "isaac_bimanual_carry",
    reset_env: bool = True,
    use_stamped: bool = True,
) -> None:
    """Run default IsaacSim bimanual carry flow with robot-specific config."""
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
        gripper_open, gripper_closed = _resolve_gripper_open_close(
            robot_cfg.gripper_control_mode
        )
        frame_id = robot_cfg.base_link_entity_path.rsplit("/", 1)[-1] if use_stamped else "arm_base"

        if reset_env:
            reset_simulation_and_randomize_object(
                carry_task_cfg.source_object_entity_path,
                xyz_offset=carry_task_cfg.object_xyz_random_offset,
                post_reset_wait=robot_cfg.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )

        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(robot_cfg.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)

        obs0 = robot.get_observation()
        left_home_pose = obs_to_pose(obs0, "left_ee")
        right_home_pose = obs_to_pose(obs0, "right_ee")

        # Pre-carry: open both grippers at home position.
        # Use the frame_id from the ee pose topic (observation frame).
        ee_frame_id = (
            (robot.ros2_interface.left_arm_handler.frame_id
             if robot.ros2_interface.left_arm_handler else None)
            or frame_id
        )
        pregrasp_sequence = [
            StageTarget(
                name="pregrasp",
                left=ArmTarget(pose=left_home_pose, gripper=gripper_open),
                right=ArmTarget(pose=right_home_pose, gripper=gripper_open),
            )
        ]
        execute_stage_sequence(
            interface=robot.ros2_interface,
            sequence=pregrasp_sequence,
            send_mode=SendMode.DUAL_ARM_STAMPED if use_stamped else SendMode.UNSTAMPED,
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
        object_center = get_object_pose_from_service(
            base_world_pos,
            base_world_quat,
            carry_task_cfg.source_object_entity_path,
            include_orientation=False,
        )

        carry_sequence = build_bimanual_carry_record_sequence(
            carry_task_cfg=carry_task_cfg,
            object_center=object_center,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
        )

        execute_stage_sequence(
            interface=robot.ros2_interface,
            sequence=carry_sequence,
            send_mode=SendMode.DUAL_ARM_STAMPED if use_stamped else SendMode.UNSTAMPED,
            frame_id=frame_id,
            arrival_timeout=robot_cfg.arrival_timeout,
            arrival_poll=robot_cfg.arrival_poll,
            time_now_fn=sim_time.now_seconds,
            sleep_fn=sim_time.sleep,
            gripper_action_wait=robot_cfg.gripper_action_wait,
            warn_prefix="Carry stage timeout",
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
        print("[OK] Bimanual carry demo completed")
    except Exception as err:
        print(f"[ERROR] Demo failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")
