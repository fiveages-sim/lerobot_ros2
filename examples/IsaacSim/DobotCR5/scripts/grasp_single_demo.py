#!/usr/bin/env python3
"""
ROS2 Grasp Demo using the LeRobot standard interface.

python examples/IsaacSim_DobotCR5/scripts/grasp_single_demo.py

This script resets simulation, optionally randomizes object x/y, queries object
and base_link world poses from ROS2 services, converts object pose into the
base frame, then executes a grasp-transport-release-return sequence.
"""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Pose

from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
    build_single_arm_grasp_transport_release_sequence,
)
from grasp_config import GRASP_CFG
build_grasp_transport_release_sequence = build_single_arm_grasp_transport_release_sequence

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))

from isaac_ros2_sim_common import (
    SimTimeHelper,
    action_from_pose,
    get_entity_pose_world_service,
    get_object_pose_from_service,
    reset_simulation_and_randomize_object,
)

# ---------------------------------------------------------------------------
# Configuration parameters
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def build_robot_config() -> ROS2RobotConfig:
    """Create a ROS2 robot configuration tailored for grasping."""
    return ROS2RobotConfig(
        id="ros2_grasp_robot",
        ros2_interface=ROS2RobotInterfaceConfig(
            joint_states_topic="/joint_states",
            end_effector_pose_topic="/left_current_pose",
            end_effector_target_topic="/left_target",
            control_type=ControlType.CARTESIAN_POSE,
            max_linear_velocity=0.1,
            max_angular_velocity=0.5,
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            min_joint_positions=[-3.14] * 6,
            max_joint_positions=[3.14] * 6,
            joint_state_timeout=1.0,
            end_effector_pose_timeout=1.0,
            gripper_enabled=True,
            gripper_joint_name="gripper_joint",
            gripper_min_position=0.0,
            gripper_max_position=1.0,
            gripper_command_topic="gripper_joint/position_command",
        ),
    )


def main() -> None:
    print("ROS2 Grasp Demo (LeRobot Standard Interface)")
    print("=" * 70)
    print("This demo will:")
    print("1. Connect to robot, then reset simulation state.")
    print("2. Query object/base world pose from get_entity_state and switch FSM to HOLD then OCS2.")
    print("3. Grasp object, transport to shelf, release, and return home.")
    print("Press Ctrl+C to abort at any time.")
    print("-" * 70)

    robot_config = build_robot_config()
    robot = ROS2Robot(robot_config)
    sim_time = SimTimeHelper()

    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        print("\nInterrupt received, shutting down...")
        try:
            if robot_connected:
                robot.disconnect()
        finally:
            sim_time.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting to robot...")
        robot.connect()
        robot_connected = True
        print("[OK] Robot connected")

        print("Resetting simulation state...")
        reset_simulation_and_randomize_object(
            GRASP_CFG.shared.object_entity_path,
            xyz_offset=GRASP_CFG.runtime.object_xyz_random_offset,
            post_reset_wait=GRASP_CFG.runtime.post_reset_wait,
            sleep_fn=sim_time.sleep,
        )

        print("Switching FSM: HOLD -> OCS2 ...")
        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_hold)
        sim_time.sleep(GRASP_CFG.runtime.fsm_switch_delay)
        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_ocs2)
        print("[OK] FSM switched to OCS2")

        initial_obs = robot.get_observation()
        initial_pose = Pose()
        initial_pose.position.x = initial_obs["left_ee.pos.x"]
        initial_pose.position.y = initial_obs["left_ee.pos.y"]
        initial_pose.position.z = initial_obs["left_ee.pos.z"]
        initial_pose.orientation.x = initial_obs["left_ee.quat.x"]
        initial_pose.orientation.y = initial_obs["left_ee.quat.y"]
        initial_pose.orientation.z = initial_obs["left_ee.quat.z"]
        initial_pose.orientation.w = initial_obs["left_ee.quat.w"]

        print("[Info] Querying base/object pose once before grasp...")
        base_world_pos, base_world_quat = get_entity_pose_world_service(
            GRASP_CFG.runtime.base_link_entity_path,
        )
        print(
            f"[OK] base_link world position: "
            f"x={base_world_pos[0]:.6f}, y={base_world_pos[1]:.6f}, z={base_world_pos[2]:.6f}"
        )
        print(f"Getting object pose from entity state: {GRASP_CFG.shared.object_entity_path}")
        pose = get_object_pose_from_service(
            base_world_pos,
            base_world_quat,
            GRASP_CFG.shared.object_entity_path,
            include_orientation=GRASP_CFG.shared.use_object_orientation,
        )
        print(
            f"[OK] Object pose (base frame): "
            f"x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}"
        )
        current_obs = robot.get_observation()
        current_orientation = (
            current_obs["left_ee.quat.x"],
            current_obs["left_ee.quat.y"],
            current_obs["left_ee.quat.z"],
            current_obs["left_ee.quat.w"],
        )
        if GRASP_CFG.shared.use_object_orientation:
            orientation_vec = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
            if np.linalg.norm(orientation_vec) < 1e-3:
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ) = current_orientation
        else:
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = current_orientation

        sequence = build_grasp_transport_release_sequence(
            pose,
            approach_clearance=GRASP_CFG.motion.approach_clearance,
            grasp_clearance=GRASP_CFG.motion.grasp_clearance,
            grasp_orientation=GRASP_CFG.motion.grasp_orientation,
            place_orientation=GRASP_CFG.motion.place_orientation,
            release_position=GRASP_CFG.motion.release_position,
            transport_height=GRASP_CFG.motion.transport_height,
            retract_offset_y=GRASP_CFG.motion.retract_offset_y,
            gripper_open=GRASP_CFG.motion.gripper_open,
            gripper_closed=GRASP_CFG.motion.gripper_closed,
            home_action=action_from_pose(initial_pose, GRASP_CFG.motion.gripper_closed),
        )
        stage_messages = {
            "1-Approach": "Opening gripper and moving to approach pose...",
            "2-Descend": "Descending towards object...",
            "3-Grasp": "Closing gripper to grasp object...",
            "4-Lift": "Lifting object...",
            "5-Transport": "Transporting object to shelf...",
            "6-Lower": "Lowering object onto shelf...",
            "7-Release": "Releasing object...",
            "8-Retract": "Retracting after release...",
            "9-ReturnHome": "Returning to initial pose...",
        }
        for stage_name, action in sequence:
            print(stage_messages.get(stage_name, f"Executing stage {stage_name}..."))
            robot.send_action(action)
            if stage_name == "9-ReturnHome":
                arrive_result = robot.ros2_interface.wait_until_arrive(
                    part="arm",
                    timeout=GRASP_CFG.single.return_pose_timeout,
                    poll_period=GRASP_CFG.single.return_check_period,
                    position_threshold=GRASP_CFG.single.return_pos_tol,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
            elif stage_name in {"3-Grasp", "7-Release"}:
                # Gripper position can stay off-target when holding an object.
                # Use arm arrival + short gripper settle wait instead of gripper-position arrival.
                arrive_result = robot.ros2_interface.wait_until_arrive(
                    part="arm",
                    timeout=GRASP_CFG.single.stage_arrival_timeout,
                    poll_period=GRASP_CFG.single.stage_check_period,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
                sim_time.sleep(GRASP_CFG.single.gripper_action_wait)
            else:
                arrive_result = robot.ros2_interface.wait_until_arrive(
                    part="arm",
                    timeout=GRASP_CFG.single.stage_arrival_timeout,
                    poll_period=GRASP_CFG.single.stage_check_period,
                    time_now_fn=sim_time.now_seconds,
                    sleep_fn=sim_time.sleep,
                )
            if not arrive_result.get("arrived", False):
                if stage_name == "9-ReturnHome":
                    print(
                        f"[WARN] Return-to-home timeout after {GRASP_CFG.single.return_pose_timeout:.1f}s "
                        f"(elapsed={arrive_result.get('elapsed', 0.0):.2f}s, "
                        f"pos_tol={GRASP_CFG.single.return_pos_tol:.3f}m), continue anyway."
                    )
                else:
                    print(
                        f"[WARN] Stage {stage_name} arrival timeout "
                        f"({arrive_result.get('elapsed', 0.0):.2f}s), fallback wait."
                    )
                    # Keep a short fallback to absorb delayed controller updates.
                    sim_time.sleep(GRASP_CFG.single.stage_fallback_wait)

        print("Switching FSM to HOLD...")
        robot.ros2_interface.send_fsm_command(GRASP_CFG.runtime.fsm_hold)
        print("[OK] FSM switched to HOLD")

        print("[OK] Grasp + transport + return sequence completed.")

    except Exception as err:
        print(f"[ERROR] Grasp demo failed: {err}")
    finally:
        print("Cleaning up resources...")
        sim_time.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


if __name__ == "__main__":
    main()
