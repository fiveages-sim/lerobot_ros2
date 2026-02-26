#!/usr/bin/env python3
"""Bimanual grasp demo for Agibot G1 in IsaacSim."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from geometry_msgs.msg import Pose

from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
    build_single_arm_grasp_transport_release_sequence,
    compose_bimanual_synchronized_sequence,
)
from grasp_config import GRASP_CFG

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))

from isaac_ros2_sim_common import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    SimTimeHelper,
    action_from_pose,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
)


def make_robot() -> ROS2Robot:
    cfg = ROS2RobotInterfaceConfig(
        joint_states_topic=GRASP_CFG.runtime.joint_states_topic,
        end_effector_pose_topic=GRASP_CFG.runtime.left_current_pose_topic,
        end_effector_target_topic=GRASP_CFG.runtime.left_target_topic,
        right_end_effector_pose_topic=GRASP_CFG.runtime.right_current_pose_topic,
        right_end_effector_target_topic=GRASP_CFG.runtime.right_target_topic,
        end_effector_current_target_topic=GRASP_CFG.runtime.left_current_target_topic,
        right_end_effector_current_target_topic=GRASP_CFG.runtime.right_current_target_topic,
        joint_names=list(GRASP_CFG.runtime.joint_names),
        gripper_enabled=True,
        gripper_joint_name=GRASP_CFG.runtime.left_gripper_joint_name,
        gripper_command_topic=GRASP_CFG.runtime.left_gripper_command_topic,
        right_gripper_command_topic=GRASP_CFG.runtime.right_gripper_command_topic,
        control_type=ControlType.CARTESIAN_POSE,
    )
    return ROS2Robot(
        ROS2RobotConfig(
            id="agibot_g1_bimanual_grasp",
            ros2_interface=cfg,
            gripper_control_mode=GRASP_CFG.runtime.gripper_control_mode,
        )
    )


def _obs_to_pose(obs: dict[str, float], arm_prefix: str) -> Pose:
    pose = Pose()
    pose.position.x = obs[f"{arm_prefix}.pos.x"]
    pose.position.y = obs[f"{arm_prefix}.pos.y"]
    pose.position.z = obs[f"{arm_prefix}.pos.z"]
    pose.orientation.x = obs[f"{arm_prefix}.quat.x"]
    pose.orientation.y = obs[f"{arm_prefix}.quat.y"]
    pose.orientation.z = obs[f"{arm_prefix}.quat.z"]
    pose.orientation.w = obs[f"{arm_prefix}.quat.w"]
    return pose


def _resolve_target_pose(base_pos, base_quat, object_entity_path: str) -> Pose:
    return get_object_pose_from_service(base_pos, base_quat, object_entity_path, include_orientation=False)


def _make_pose(position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
    return pose


def main() -> None:
    robot = make_robot()
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

        if not GRASP_CFG.single.right_grasp_only:
            reset_simulation_and_randomize_object(
                GRASP_CFG.scene.left_object_entity_path,
                xyz_offset=GRASP_CFG.runtime.object_xyz_random_offset,
                post_reset_wait=GRASP_CFG.runtime.post_reset_wait,
                sleep_fn=sim_time.sleep,
            )
        reset_simulation_and_randomize_object(
            GRASP_CFG.scene.right_object_entity_path,
            xyz_offset=GRASP_CFG.runtime.object_xyz_random_offset,
            post_reset_wait=GRASP_CFG.runtime.post_reset_wait,
            sleep_fn=sim_time.sleep,
        )

        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_hold)
        sim_time.sleep(GRASP_CFG.runtime.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_ocs2)

        obs0 = robot.get_observation()
        left_home_pose = _obs_to_pose(obs0, "left_ee")
        right_home_pose = _obs_to_pose(obs0, "right_ee")

        # Explicitly open gripper(s) before approaching the object.
        pregrasp_action = {
            "right_ee.pos.x": right_home_pose.position.x,
            "right_ee.pos.y": right_home_pose.position.y,
            "right_ee.pos.z": right_home_pose.position.z,
            "right_ee.quat.x": right_home_pose.orientation.x,
            "right_ee.quat.y": right_home_pose.orientation.y,
            "right_ee.quat.z": right_home_pose.orientation.z,
            "right_ee.quat.w": right_home_pose.orientation.w,
            "right_gripper.pos": GRASP_CFG.motion.gripper_open,
        }
        if not GRASP_CFG.single.right_grasp_only:
            pregrasp_action.update(
                {
                    "left_ee.pos.x": left_home_pose.position.x,
                    "left_ee.pos.y": left_home_pose.position.y,
                    "left_ee.pos.z": left_home_pose.position.z,
                    "left_ee.quat.x": left_home_pose.orientation.x,
                    "left_ee.quat.y": left_home_pose.orientation.y,
                    "left_ee.quat.z": left_home_pose.orientation.z,
                    "left_ee.quat.w": left_home_pose.orientation.w,
                    "left_gripper.pos": GRASP_CFG.motion.gripper_open,
                }
            )
        robot.send_action(pregrasp_action)
        sim_time.sleep(GRASP_CFG.single.gripper_action_wait)

        base_world_pos, base_world_quat = get_entity_pose_world_service(
            GRASP_CFG.scene.base_link_entity_path
        )
        right_target_pose = _resolve_target_pose(
            base_world_pos,
            base_world_quat,
            GRASP_CFG.scene.right_object_entity_path,
        )

        if GRASP_CFG.single.right_grasp_only:
            right_only_sequence = build_single_arm_grasp_transport_release_sequence(
                right_target_pose,
                approach_clearance=GRASP_CFG.motion.approach_clearance,
                grasp_clearance=GRASP_CFG.motion.grasp_clearance,
                grasp_orientation=GRASP_CFG.motion.grasp_orientation,
                place_orientation=(
                    right_home_pose.orientation.x,
                    right_home_pose.orientation.y,
                    right_home_pose.orientation.z,
                    right_home_pose.orientation.w,
                ),
                release_position=GRASP_CFG.motion.right_release_position,
                transport_height=GRASP_CFG.motion.transport_height,
                retract_offset_y=GRASP_CFG.motion.right_retract_offset_y,
                gripper_open=GRASP_CFG.motion.gripper_open,
                gripper_closed=GRASP_CFG.motion.gripper_closed,
                ee_prefix="right_ee",
                gripper_key="right_gripper.pos",
                home_action=action_from_pose(
                    right_home_pose,
                    GRASP_CFG.motion.gripper_closed,
                    ee_prefix="right_ee",
                    gripper_key="right_gripper.pos",
                ),
            )
            sequence = right_only_sequence
        else:
            left_target_pose = _resolve_target_pose(
                base_world_pos,
                base_world_quat,
                GRASP_CFG.scene.left_object_entity_path,
            )
            left_sequence = build_single_arm_grasp_transport_release_sequence(
                left_target_pose,
                approach_clearance=GRASP_CFG.motion.approach_clearance,
                grasp_clearance=GRASP_CFG.motion.grasp_clearance,
                grasp_orientation=GRASP_CFG.motion.grasp_orientation,
                place_orientation=(
                    left_home_pose.orientation.x,
                    left_home_pose.orientation.y,
                    left_home_pose.orientation.z,
                    left_home_pose.orientation.w,
                ),
                release_position=GRASP_CFG.motion.left_release_position,
                transport_height=GRASP_CFG.motion.transport_height,
                retract_offset_y=GRASP_CFG.motion.left_retract_offset_y,
                gripper_open=GRASP_CFG.motion.gripper_open,
                gripper_closed=GRASP_CFG.motion.gripper_closed,
                ee_prefix="left_ee",
                gripper_key="left_gripper.pos",
                home_action=action_from_pose(
                    left_home_pose,
                    GRASP_CFG.motion.gripper_closed,
                    ee_prefix="left_ee",
                    gripper_key="left_gripper.pos",
                ),
            )
            right_sequence = build_single_arm_grasp_transport_release_sequence(
                right_target_pose,
                approach_clearance=GRASP_CFG.motion.approach_clearance,
                grasp_clearance=GRASP_CFG.motion.grasp_clearance,
                grasp_orientation=GRASP_CFG.motion.grasp_orientation,
                place_orientation=(
                    right_home_pose.orientation.x,
                    right_home_pose.orientation.y,
                    right_home_pose.orientation.z,
                    right_home_pose.orientation.w,
                ),
                release_position=GRASP_CFG.motion.right_release_position,
                transport_height=GRASP_CFG.motion.transport_height,
                retract_offset_y=GRASP_CFG.motion.right_retract_offset_y,
                gripper_open=GRASP_CFG.motion.gripper_open,
                gripper_closed=GRASP_CFG.motion.gripper_closed,
                ee_prefix="right_ee",
                gripper_key="right_gripper.pos",
                home_action=action_from_pose(
                    right_home_pose,
                    GRASP_CFG.motion.gripper_closed,
                    ee_prefix="right_ee",
                    gripper_key="right_gripper.pos",
                ),
            )
            sequence = compose_bimanual_synchronized_sequence(left_sequence, right_sequence)
        if GRASP_CFG.single.stop_after_grasp:
            grasp_idx = next((i for i, (name, _) in enumerate(sequence) if "Grasp" in name), None)
            if grasp_idx is None:
                raise RuntimeError("No grasp stage found in generated sequence.")
            sequence = sequence[: grasp_idx + 1]
            print(f"[Info] stop_after_grasp=True, truncating sequence at stage: {sequence[-1][0]}")
        elif GRASP_CFG.single.stop_after_lift:
            lift_idx = next((i for i, (name, _) in enumerate(sequence) if "Lift" in name), None)
            if lift_idx is None:
                raise RuntimeError("No lift stage found in generated sequence.")
            sequence = sequence[: lift_idx + 1]
            print(f"[Info] stop_after_lift=True, truncating sequence at stage: {sequence[-1][0]}")

        for stage_name, action in sequence:
            print(f"[Stage] {stage_name}")
            robot.send_action(action)
            if GRASP_CFG.single.right_grasp_only:
                right_arrive = robot.ros2_interface.wait_until_arrive(
                    part="right_arm",
                    timeout=GRASP_CFG.single.arrival_timeout,
                    poll_period=GRASP_CFG.single.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
                left_arrive = {"arrived": True}
            else:
                left_arrive = robot.ros2_interface.wait_until_arrive(
                    part="left_arm",
                    timeout=GRASP_CFG.single.arrival_timeout,
                    poll_period=GRASP_CFG.single.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
                right_arrive = robot.ros2_interface.wait_until_arrive(
                    part="right_arm",
                    timeout=GRASP_CFG.single.arrival_timeout,
                    poll_period=GRASP_CFG.single.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
            if "Grasp" in stage_name or "Release" in stage_name:
                sim_time.sleep(GRASP_CFG.single.gripper_action_wait)
            if not left_arrive.get("arrived", False) or not right_arrive.get("arrived", False):
                print(
                    "[WARN] Stage timeout: "
                    f"left_arrived={left_arrive.get('arrived', False)}, "
                    f"right_arrived={right_arrive.get('arrived', False)}"
                )

        # After right-arm grasp pipeline, move to handover pose and let left arm grasp.
        if GRASP_CFG.single.right_grasp_only:
            handover_pose = _make_pose(
                GRASP_CFG.motion.handover_position,
                GRASP_CFG.motion.handover_orientation,
            )
            left_handover_pose = _make_pose(
                GRASP_CFG.motion.handover_position,
                GRASP_CFG.motion.left_handover_orientation,
            )
            left_handover_pose.position.z -= 0.02
            left_handover_pose.position.y -= 0.02
            left_place_pose = _make_pose(
                GRASP_CFG.motion.left_place_position,
                GRASP_CFG.motion.left_place_orientation,
            )
            handover_sequence = [
                (
                    "Handover-1-SyncMove",
                    {
                        "left_ee.pos.x": left_handover_pose.position.x,
                        "left_ee.pos.y": left_handover_pose.position.y,
                        "left_ee.pos.z": left_handover_pose.position.z,
                        "left_ee.quat.x": left_handover_pose.orientation.x,
                        "left_ee.quat.y": left_handover_pose.orientation.y,
                        "left_ee.quat.z": left_handover_pose.orientation.z,
                        "left_ee.quat.w": left_handover_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_open,
                        "right_ee.pos.x": handover_pose.position.x,
                        "right_ee.pos.y": handover_pose.position.y,
                        "right_ee.pos.z": handover_pose.position.z,
                        "right_ee.quat.x": handover_pose.orientation.x,
                        "right_ee.quat.y": handover_pose.orientation.y,
                        "right_ee.quat.z": handover_pose.orientation.z,
                        "right_ee.quat.w": handover_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_closed,
                    },
                ),
                (
                    "Handover-3-LeftGrasp",
                    {
                        "left_ee.pos.x": left_handover_pose.position.x,
                        "left_ee.pos.y": left_handover_pose.position.y,
                        "left_ee.pos.z": left_handover_pose.position.z,
                        "left_ee.quat.x": left_handover_pose.orientation.x,
                        "left_ee.quat.y": left_handover_pose.orientation.y,
                        "left_ee.quat.z": left_handover_pose.orientation.z,
                        "left_ee.quat.w": left_handover_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_closed,
                        "right_ee.pos.x": handover_pose.position.x,
                        "right_ee.pos.y": handover_pose.position.y,
                        "right_ee.pos.z": handover_pose.position.z,
                        "right_ee.quat.x": handover_pose.orientation.x,
                        "right_ee.quat.y": handover_pose.orientation.y,
                        "right_ee.quat.z": handover_pose.orientation.z,
                        "right_ee.quat.w": handover_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_closed,
                    },
                ),
                (
                    "Handover-4-RightRelease",
                    {
                        "left_ee.pos.x": left_handover_pose.position.x,
                        "left_ee.pos.y": left_handover_pose.position.y,
                        "left_ee.pos.z": left_handover_pose.position.z,
                        "left_ee.quat.x": left_handover_pose.orientation.x,
                        "left_ee.quat.y": left_handover_pose.orientation.y,
                        "left_ee.quat.z": left_handover_pose.orientation.z,
                        "left_ee.quat.w": left_handover_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_closed,
                        "right_ee.pos.x": handover_pose.position.x,
                        "right_ee.pos.y": handover_pose.position.y,
                        "right_ee.pos.z": handover_pose.position.z,
                        "right_ee.quat.x": handover_pose.orientation.x,
                        "right_ee.quat.y": handover_pose.orientation.y,
                        "right_ee.quat.z": handover_pose.orientation.z,
                        "right_ee.quat.w": handover_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_open,
                    },
                ),
                (
                    "Handover-5-LeftPlaceAndRightHome",
                    {
                        "left_ee.pos.x": left_place_pose.position.x,
                        "left_ee.pos.y": left_place_pose.position.y,
                        "left_ee.pos.z": left_place_pose.position.z,
                        "left_ee.quat.x": left_place_pose.orientation.x,
                        "left_ee.quat.y": left_place_pose.orientation.y,
                        "left_ee.quat.z": left_place_pose.orientation.z,
                        "left_ee.quat.w": left_place_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_closed,
                        "right_ee.pos.x": right_home_pose.position.x,
                        "right_ee.pos.y": right_home_pose.position.y,
                        "right_ee.pos.z": right_home_pose.position.z,
                        "right_ee.quat.x": right_home_pose.orientation.x,
                        "right_ee.quat.y": right_home_pose.orientation.y,
                        "right_ee.quat.z": right_home_pose.orientation.z,
                        "right_ee.quat.w": right_home_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_open,
                    },
                ),
                (
                    "Handover-6-LeftReleaseAtPlace",
                    {
                        "left_ee.pos.x": left_place_pose.position.x,
                        "left_ee.pos.y": left_place_pose.position.y,
                        "left_ee.pos.z": left_place_pose.position.z,
                        "left_ee.quat.x": left_place_pose.orientation.x,
                        "left_ee.quat.y": left_place_pose.orientation.y,
                        "left_ee.quat.z": left_place_pose.orientation.z,
                        "left_ee.quat.w": left_place_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_open,
                        "right_ee.pos.x": right_home_pose.position.x,
                        "right_ee.pos.y": right_home_pose.position.y,
                        "right_ee.pos.z": right_home_pose.position.z,
                        "right_ee.quat.x": right_home_pose.orientation.x,
                        "right_ee.quat.y": right_home_pose.orientation.y,
                        "right_ee.quat.z": right_home_pose.orientation.z,
                        "right_ee.quat.w": right_home_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_open,
                    },
                ),
                (
                    "Handover-7-LeftReturnHome",
                    {
                        "left_ee.pos.x": left_home_pose.position.x,
                        "left_ee.pos.y": left_home_pose.position.y,
                        "left_ee.pos.z": left_home_pose.position.z,
                        "left_ee.quat.x": left_home_pose.orientation.x,
                        "left_ee.quat.y": left_home_pose.orientation.y,
                        "left_ee.quat.z": left_home_pose.orientation.z,
                        "left_ee.quat.w": left_home_pose.orientation.w,
                        "left_gripper.pos": GRASP_CFG.motion.gripper_open,
                        "right_ee.pos.x": right_home_pose.position.x,
                        "right_ee.pos.y": right_home_pose.position.y,
                        "right_ee.pos.z": right_home_pose.position.z,
                        "right_ee.quat.x": right_home_pose.orientation.x,
                        "right_ee.quat.y": right_home_pose.orientation.y,
                        "right_ee.quat.z": right_home_pose.orientation.z,
                        "right_ee.quat.w": right_home_pose.orientation.w,
                        "right_gripper.pos": GRASP_CFG.motion.gripper_open,
                    },
                ),
            ]
            for stage_name, action in handover_sequence:
                print(f"[Stage] {stage_name}")
                robot.send_action(action)
                left_arrive = robot.ros2_interface.wait_until_arrive(
                    part="left_arm",
                    timeout=GRASP_CFG.single.arrival_timeout,
                    poll_period=GRASP_CFG.single.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
                right_arrive = robot.ros2_interface.wait_until_arrive(
                    part="right_arm",
                    timeout=GRASP_CFG.single.arrival_timeout,
                    poll_period=GRASP_CFG.single.arrival_poll,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
                if "Grasp" in stage_name or "Release" in stage_name:
                    sim_time.sleep(GRASP_CFG.single.gripper_action_wait)
                if not left_arrive.get("arrived", False) or not right_arrive.get("arrived", False):
                    print(
                        "[WARN] Handover stage timeout: "
                        f"left_arrived={left_arrive.get('arrived', False)}, "
                        f"right_arrived={right_arrive.get('arrived', False)}"
                    )
                # Ensure left arm reaches handover approach before executing left grasp.
                if stage_name == "Handover-1-SyncMove" and not left_arrive.get("arrived", False):
                    raise RuntimeError("Left arm did not arrive at handover approach pose; skip left grasp.")

        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_hold)
        print("[OK] Bimanual grasp demo completed")
    except Exception as err:
        print(f"[ERROR] Demo failed: {err}")
    finally:
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


if __name__ == "__main__":
    main()
