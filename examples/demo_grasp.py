#!/usr/bin/env python3
"""
ROS2 Grasp Demo using the LeRobot standard interface.

python examples/demo_grasp.py

This script listens for an object pose published on /isaac/tf (relative to
base_link), approaches the object, closes the gripper, and then lifts it.
It demonstrates how to integrate LeRobot's ROS2 plugin with external pose
estimators without using a higher-level motion planner.
"""

import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
)

# ---------------------------------------------------------------------------
# Configuration parameters
# ---------------------------------------------------------------------------
OBJECT_TF_TOPIC = "/isaac/appletf"   # TF topic carrying the object transform
TARGET_FRAME_ID = "apple"      # Child frame name for the object pose

APPROACH_CLEARANCE = 0.12       # Meters above object before descending
GRASP_CLEARANCE = -0.03         # Meters above object when closing gripper
MOVE_DURATION = 3.0             # Seconds to wait after each motion command
GRIPPER_OPEN = 1.0              # Fully open gripper command
GRIPPER_CLOSED = 0.0            # Fully closed gripper command
POSE_TIMEOUT = 10.0             # Seconds to wait for TF before aborting
USE_OBJECT_ORIENTATION = False  # If False, keep current wrist orientation
# ---------------------------------------------------------------------------


@dataclass
class PoseSample:
    """Container for pose data extracted from TF."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    timestamp: float


class ObjectPoseListener:
    """Subscribe to TF updates and keep the latest pose for the target frame."""

    def __init__(self, topic: str, target_frame: str) -> None:
        self.topic = topic
        self.target_frame = target_frame
        self._latest: Optional[PoseSample] = None
        self._lock = threading.Lock()
        self._update_event = threading.Event()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("lerobot_object_pose_listener")
        self._sub = self._node.create_subscription(
            TFMessage,
            self.topic,
            self._tf_callback,
            10,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
        )
        self._thread.start()

    def _tf_callback(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id != self.target_frame:
                continue

            trans = transform.transform.translation
            rot = transform.transform.rotation
            sample = PoseSample(
                x=trans.x,
                y=trans.y,
                z=trans.z,
                qx=rot.x,
                qy=rot.y,
                qz=rot.z,
                qw=rot.w,
                timestamp=time.time(),
            )
            with self._lock:
                self._latest = sample
                self._update_event.set()
            break

    def wait_for_pose(self, timeout: float) -> Optional[PoseSample]:
        """Block until a pose sample is received or timeout expires."""
        received = self._update_event.wait(timeout)
        if not received:
            return None
        return self.get_latest()

    def get_latest(self) -> Optional[PoseSample]:
        """Return the most recent pose sample."""
        with self._lock:
            return self._latest

    def shutdown(self) -> None:
        """Stop executor and destroy ROS 2 resources."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sub is not None:
            self._sub.destroy()
            self._sub = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


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


def action_from_pose(pose: Pose, gripper: float) -> dict[str, float]:
    """Convert a pose and gripper command into a LeRobot action dictionary."""
    action = {
        "end_effector.position.x": pose.position.x,
        "end_effector.position.y": pose.position.y,
        "end_effector.position.z": pose.position.z,
        "end_effector.orientation.x": pose.orientation.x,
        "end_effector.orientation.y": pose.orientation.y,
        "end_effector.orientation.z": pose.orientation.z,
        "end_effector.orientation.w": pose.orientation.w,
    }
    action["gripper.position"] = float(np.clip(gripper, 0.0, 1.0))
    return action


def pose_from_sample(sample: PoseSample) -> Pose:
    """Build a Pose message from a PoseSample."""
    pose = Pose()
    pose.position.x = sample.x
    pose.position.y = sample.y
    pose.position.z = sample.z
    pose.orientation.x = sample.qx
    pose.orientation.y = sample.qy
    pose.orientation.z = sample.qz
    pose.orientation.w = sample.qw
    return pose


def main() -> None:
    print("ROS2 Grasp Demo (LeRobot Standard Interface)")
    print("=" * 70)
    print("This demo will:")
    print("1. Connect to the robot via LeRobot's ROS2 interface.")
    print(f"2. Listen for {TARGET_FRAME_ID} transforms on {OBJECT_TF_TOPIC}.")
    print("3. Move above the object, descend, close gripper, then lift.")
    print("Press Ctrl+C to abort at any time.")
    print("-" * 70)

    robot_config = build_robot_config()
    robot = ROS2Robot(robot_config)
    object_listener = ObjectPoseListener(OBJECT_TF_TOPIC, TARGET_FRAME_ID)

    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        print("\nInterrupt received, shutting down...")
        try:
            if robot_connected:
                robot.disconnect()
        finally:
            object_listener.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        print("Connecting to robot...")
        robot.connect()
        robot_connected = True
        print("[OK] Robot connected")

        print("Waiting for object pose from TF...")
        sample = object_listener.wait_for_pose(timeout=POSE_TIMEOUT)
        if sample is None:
            raise RuntimeError(
                f"No transform for frame '{TARGET_FRAME_ID}' within {POSE_TIMEOUT:.1f}s"
            )
        print(
            f"[OK] Received object pose at "
            f"x={sample.x:.3f}, y={sample.y:.3f}, z={sample.z:.3f}"
        )

        # Build pose from TF sample and ensure orientation is valid.
        pose = pose_from_sample(sample)
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

        pose.orientation.x=0.7077044621721283
        pose.orientation.y=0.7065080817531944
        pose.orientation.z=0.000848011543394921
        pose.orientation.w=-7.42664282289885e-05

        approach_pose = Pose()
        approach_pose.position.x = pose.position.x
        approach_pose.position.y = pose.position.y
        approach_pose.position.z = pose.position.z + APPROACH_CLEARANCE
        approach_pose.orientation.x = pose.orientation.x
        approach_pose.orientation.y = pose.orientation.y
        approach_pose.orientation.z = pose.orientation.z
        approach_pose.orientation.w = pose.orientation.w

        descend_pose = Pose()
        descend_pose.position.x = pose.position.x
        descend_pose.position.y = pose.position.y
        descend_pose.position.z = pose.position.z + GRASP_CLEARANCE
        descend_pose.orientation.x = pose.orientation.x
        descend_pose.orientation.y = pose.orientation.y
        descend_pose.orientation.z = pose.orientation.z
        descend_pose.orientation.w = pose.orientation.w

        lift_pose = Pose()
        lift_pose.position.x = pose.position.x
        lift_pose.position.y = pose.position.y
        lift_pose.position.z = pose.position.z + APPROACH_CLEARANCE
        lift_pose.orientation.x = pose.orientation.x
        lift_pose.orientation.y = pose.orientation.y
        lift_pose.orientation.z = pose.orientation.z
        lift_pose.orientation.w = pose.orientation.w

        print("Opening gripper and moving to approach pose...")
        robot.send_action(action_from_pose(approach_pose, GRIPPER_OPEN))
        time.sleep(MOVE_DURATION)

        print("Descending towards object...")
        robot.send_action(action_from_pose(descend_pose, GRIPPER_OPEN))
        time.sleep(MOVE_DURATION)

        print("Closing gripper to grasp object...")
        robot.send_action(action_from_pose(descend_pose, GRIPPER_CLOSED))
        time.sleep(1.5)

        print("Lifting object...")
        robot.send_action(action_from_pose(lift_pose, GRIPPER_CLOSED))
        time.sleep(MOVE_DURATION)

        print("[OK] Grasp sequence completed.")

    except Exception as err:
        print(f"[ERROR] Grasp demo failed: {err}")
    finally:
        print("Cleaning up resources...")
        object_listener.shutdown()
        if robot_connected:
            robot.disconnect()
            print("[OK] Robot disconnected")


if __name__ == "__main__":
    main()
