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

import numpy as np
from geometry_msgs.msg import Pose

from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
)
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
OBJECT_ENTITY_PATH = "/World/apple/apple/apple"     # Entity path for object in Isaac Sim

APPROACH_CLEARANCE = 0.12       # Meters above object before descending
GRASP_CLEARANCE = -0.03         # Meters above object when closing gripper
MOVE_DURATION = 0.6             # Seconds to wait after each motion command
GRIPPER_ACTION_WAIT = 0.6       # Seconds to wait after open/close gripper
RETURN_PAUSE = 1.0              # Seconds to wait for return-to-home moves
RETURN_POSE_TIMEOUT = 8.0       # Max seconds to wait for initial pose arrival
RETURN_POS_TOL = 0.02           # Position tolerance (m) for return pose check
RETURN_CHECK_PERIOD = 0.05      # Poll period while waiting for return pose
GRIPPER_OPEN = 1.0              # Fully open gripper command
GRIPPER_CLOSED = 0.0            # Fully closed gripper command
USE_OBJECT_ORIENTATION = False  # If False, keep current wrist orientation

# Grasp orientation (vertical-down) for approach/descend/lift stages
GRASP_ORIENTATION_X = 0.7
GRASP_ORIENTATION_Y = 0.7
GRASP_ORIENTATION_Z = 0.00
GRASP_ORIENTATION_W = 0.00

# Shelf release configuration (from auto_grasp_record transport/release flow)
RELEASE_POSITION_X = -0.5011657941743888
RELEASE_POSITION_Y = 0.36339369774887115
RELEASE_POSITION_Z = 0.34
TRANSPORT_HEIGHT = 0.4
RETRACT_OFFSET_Y = -0.2
# Placement orientation (used for transport/lower/retract stages)
PLACE_ORIENTATION_X = 0.64
PLACE_ORIENTATION_Y = 0.64
PLACE_ORIENTATION_Z = -0.28
PLACE_ORIENTATION_W = -0.28
FSM_HOLD = 2
FSM_OCS2 = 3
FSM_SWITCH_DELAY = 0.1
POST_RESET_WAIT = 1.0
SIM_STATE_PLAYING = 1
SIM_STATE_RESET = 0
SIM_SERVICE_CALL_TIMEOUT = 8.0
ENTITY_STATE_SERVICE_TIMEOUT = 8.0
BASE_LINK_ENTITY_PATH = "/World/DobotCR5_ROS2/DobotCR5/base_link"
PRIM_ATTR_SERVICE_TIMEOUT = 8.0
ENABLE_OBJECT_XY_RANDOMIZATION = True
OBJECT_XY_RANDOM_OFFSET = 0.5
ROS_WAIT_POLL_PERIOD = 0.01
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


def wait_for_position(
    robot: ROS2Robot,
    time_source: SimTimeHelper,
    target_pose: Pose,
    timeout: float,
    pos_tolerance: float,
) -> bool:
    """Wait until end-effector position is close to target pose."""
    start_time = time_source.now_seconds()
    wall_start = time.monotonic()
    while (time_source.now_seconds() - start_time) <= timeout:
        if (time.monotonic() - wall_start) > max(timeout + 2.0, 5.0):
            print("[WARN] wait_for_position wall-time guard triggered; stop waiting.")
            return False
        obs = robot.get_observation()
        dx = obs["end_effector.position.x"] - target_pose.position.x
        dy = obs["end_effector.position.y"] - target_pose.position.y
        dz = obs["end_effector.position.z"] - target_pose.position.z
        pos_err = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if pos_err <= pos_tolerance:
            return True
        time_source.sleep(RETURN_CHECK_PERIOD)
    return False


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
            OBJECT_ENTITY_PATH,
            sim_state_reset=SIM_STATE_RESET,
            sim_state_playing=SIM_STATE_PLAYING,
            sim_service_timeout=SIM_SERVICE_CALL_TIMEOUT,
            enable_randomization=ENABLE_OBJECT_XY_RANDOMIZATION,
            xy_offset=OBJECT_XY_RANDOM_OFFSET,
            prim_attr_timeout=PRIM_ATTR_SERVICE_TIMEOUT,
            post_reset_wait=POST_RESET_WAIT,
            sleep_fn=sim_time.sleep,
        )

        print("Switching FSM: HOLD -> OCS2 ...")
        robot.ros2_interface.send_fsm_command(FSM_HOLD)
        sim_time.sleep(FSM_SWITCH_DELAY)
        robot.ros2_interface.send_fsm_command(FSM_OCS2)
        print("[OK] FSM switched to OCS2")

        initial_obs = robot.get_observation()
        initial_pose = Pose()
        initial_pose.position.x = initial_obs["end_effector.position.x"]
        initial_pose.position.y = initial_obs["end_effector.position.y"]
        initial_pose.position.z = initial_obs["end_effector.position.z"]
        initial_pose.orientation.x = initial_obs["end_effector.orientation.x"]
        initial_pose.orientation.y = initial_obs["end_effector.orientation.y"]
        initial_pose.orientation.z = initial_obs["end_effector.orientation.z"]
        initial_pose.orientation.w = initial_obs["end_effector.orientation.w"]

        print("[Info] Querying base/object pose once before grasp...")
        base_world_pos, base_world_quat = get_entity_pose_world_service(
            BASE_LINK_ENTITY_PATH,
            timeout=ENTITY_STATE_SERVICE_TIMEOUT,
        )
        print(
            f"[OK] base_link world position: "
            f"x={base_world_pos[0]:.6f}, y={base_world_pos[1]:.6f}, z={base_world_pos[2]:.6f}"
        )
        print(f"Getting object pose from entity state: {OBJECT_ENTITY_PATH}")
        pose = get_object_pose_from_service(
            base_world_pos,
            base_world_quat,
            OBJECT_ENTITY_PATH,
            include_orientation=USE_OBJECT_ORIENTATION,
            entity_state_timeout=ENTITY_STATE_SERVICE_TIMEOUT,
        )
        print(
            f"[OK] Object pose (base frame): "
            f"x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}"
        )
        current_obs = robot.get_observation()
        current_orientation = (
            current_obs["end_effector.orientation.x"],
            current_obs["end_effector.orientation.y"],
            current_obs["end_effector.orientation.z"],
            current_obs["end_effector.orientation.w"],
        )
        if USE_OBJECT_ORIENTATION:
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

        approach_pose = Pose()
        approach_pose.position.x = pose.position.x
        approach_pose.position.y = pose.position.y
        approach_pose.position.z = pose.position.z + APPROACH_CLEARANCE
        approach_pose.orientation.x = GRASP_ORIENTATION_X
        approach_pose.orientation.y = GRASP_ORIENTATION_Y
        approach_pose.orientation.z = GRASP_ORIENTATION_Z
        approach_pose.orientation.w = GRASP_ORIENTATION_W

        descend_pose = Pose()
        descend_pose.position.x = pose.position.x
        descend_pose.position.y = pose.position.y
        descend_pose.position.z = pose.position.z + GRASP_CLEARANCE
        descend_pose.orientation.x = GRASP_ORIENTATION_X
        descend_pose.orientation.y = GRASP_ORIENTATION_Y
        descend_pose.orientation.z = GRASP_ORIENTATION_Z
        descend_pose.orientation.w = GRASP_ORIENTATION_W

        lift_pose = Pose()
        lift_pose.position.x = pose.position.x
        lift_pose.position.y = pose.position.y
        lift_pose.position.z = pose.position.z + APPROACH_CLEARANCE
        lift_pose.orientation.x = GRASP_ORIENTATION_X
        lift_pose.orientation.y = GRASP_ORIENTATION_Y
        lift_pose.orientation.z = GRASP_ORIENTATION_Z
        lift_pose.orientation.w = GRASP_ORIENTATION_W

        transport_pose = Pose()
        transport_pose.position.x = RELEASE_POSITION_X
        transport_pose.position.y = RELEASE_POSITION_Y
        transport_pose.position.z = TRANSPORT_HEIGHT
        transport_pose.orientation.x = PLACE_ORIENTATION_X
        transport_pose.orientation.y = PLACE_ORIENTATION_Y
        transport_pose.orientation.z = PLACE_ORIENTATION_Z
        transport_pose.orientation.w = PLACE_ORIENTATION_W

        lower_pose = Pose()
        lower_pose.position.x = RELEASE_POSITION_X
        lower_pose.position.y = RELEASE_POSITION_Y
        lower_pose.position.z = RELEASE_POSITION_Z
        lower_pose.orientation.x = PLACE_ORIENTATION_X
        lower_pose.orientation.y = PLACE_ORIENTATION_Y
        lower_pose.orientation.z = PLACE_ORIENTATION_Z
        lower_pose.orientation.w = PLACE_ORIENTATION_W

        retract_pose = Pose()
        retract_pose.position.x = RELEASE_POSITION_X
        retract_pose.position.y = RELEASE_POSITION_Y + RETRACT_OFFSET_Y
        retract_pose.position.z = RELEASE_POSITION_Z
        retract_pose.orientation.x = PLACE_ORIENTATION_X
        retract_pose.orientation.y = PLACE_ORIENTATION_Y
        retract_pose.orientation.z = PLACE_ORIENTATION_Z
        retract_pose.orientation.w = PLACE_ORIENTATION_W

        print("Opening gripper and moving to approach pose...")
        robot.send_action(action_from_pose(approach_pose, GRIPPER_OPEN))
        sim_time.sleep(MOVE_DURATION)

        print("Descending towards object...")
        robot.send_action(action_from_pose(descend_pose, GRIPPER_OPEN))
        sim_time.sleep(MOVE_DURATION)

        print("Closing gripper to grasp object...")
        robot.send_action(action_from_pose(descend_pose, GRIPPER_CLOSED))
        sim_time.sleep(GRIPPER_ACTION_WAIT)

        print("Lifting object...")
        robot.send_action(action_from_pose(lift_pose, GRIPPER_CLOSED))
        sim_time.sleep(MOVE_DURATION)

        print("Transporting object to shelf...")
        robot.send_action(action_from_pose(transport_pose, GRIPPER_CLOSED))
        sim_time.sleep(MOVE_DURATION)

        print("Lowering object onto shelf...")
        robot.send_action(action_from_pose(lower_pose, GRIPPER_CLOSED))
        sim_time.sleep(MOVE_DURATION)

        print("Releasing object...")
        robot.send_action(action_from_pose(lower_pose, GRIPPER_OPEN))
        sim_time.sleep(GRIPPER_ACTION_WAIT)

        print("Retracting after release...")
        robot.send_action(action_from_pose(retract_pose, GRIPPER_OPEN))
        sim_time.sleep(MOVE_DURATION)

        print("Returning to initial pose...")
        robot.send_action(action_from_pose(initial_pose, GRIPPER_OPEN))
        arrived_home = wait_for_position(
            robot,
            sim_time,
            initial_pose,
            timeout=RETURN_POSE_TIMEOUT,
            pos_tolerance=RETURN_POS_TOL,
        )
        if not arrived_home:
            print(
                f"[WARN] Return-to-home timeout after {RETURN_POSE_TIMEOUT:.1f}s "
                f"(tol={RETURN_POS_TOL:.3f}m), continue anyway."
            )
            sim_time.sleep(RETURN_PAUSE)

        print("Switching FSM to HOLD...")
        robot.ros2_interface.send_fsm_command(FSM_HOLD)
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
