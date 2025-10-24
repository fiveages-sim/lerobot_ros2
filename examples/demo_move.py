#!/usr/bin/env python3
"""
Simple ROS2 Robot Control Example using LeRobot Standard Interface

This script demonstrates basic ROS2 robot control using LeRobot's standard
get_observation() and send_action() interfaces. It records the current 
end-effector position and then cycles between the current position and a 
target position (x-axis +0.1).

This example shows how to use the standard LeRobot robot interface instead
of directly accessing ROS2 topics, making it compatible with the broader
LeRobot ecosystem.

Usage:
    python simple_ros2_control.py

Requirements:
    - ROS2 environment must be sourced
    - ROS2 robot must be running and publishing joint states
    - lerobot_robot_ros2 and lerobot_camera_ros2 packages must be installed
"""

import time
import signal
import sys
import numpy as np

# Import our custom ROS2 plugins
from lerobot_robot_ros2 import ROS2RobotConfig, ROS2Robot, ROS2RobotInterfaceConfig, ControlType


def main():
    """Simple ROS2 robot control example with gripper control using LeRobot standard interface."""
    print("Simple ROS2 Robot Control Example with Gripper Control (LeRobot Standard Interface)")
    print("=" * 70)
    print("This script will:")
    print("1. Get current robot observation using get_observation()")
    print("2. Send target action using send_action() (x + 0.1) with gripper OPEN")
    print("3. Send current action (back to original position) with gripper CLOSED")
    print("4. Cycle between these two positions with gripper open/close")
    print("5. Monitor robot state and gripper position with get_observation()")
    print("Press Ctrl+C to stop")
    print("-" * 70)
    
    # Create robot configuration (no cameras needed for pure motion)
    robot_config = ROS2RobotConfig(
        id="simple_ros2_robot",
        ros2_interface=ROS2RobotInterfaceConfig(
            joint_states_topic="/joint_states",
            end_effector_pose_topic="/left_current_pose",
            end_effector_target_topic="/left_target",
            control_type=ControlType.CARTESIAN_POSE,
            max_linear_velocity=0.1,
            max_angular_velocity=0.5,
            joint_names=["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6"],
            min_joint_positions=[-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
            max_joint_positions=[3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
            joint_state_timeout=1.0,
            end_effector_pose_timeout=1.0,
            # Gripper configuration
            gripper_enabled=True,
            gripper_joint_name="left_gripper_joint",
            gripper_min_position=0.0,  # Closed position
            gripper_max_position=1.0   # Open position
        )
    )
    
    # Create robot instance
    robot = ROS2Robot(robot_config)
    
    # Connect robot
    print("Connecting robot...")
    robot.connect()
    print("✓ Robot connected")
    
    # Wait for robot to be ready
    print("Waiting for robot to be ready...")
    time.sleep(2)
    
    # Display robot features
    print("\n" + "="*60)
    print("ROBOT FEATURES INFORMATION")
    print("="*60)
    
    # Display observation features
    print("Observation Features:")
    obs_features = robot.observation_features
    for key, value in obs_features.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal observation features: {len(obs_features)}")
    
    # Display action features
    print("\nAction Features:")
    action_features = robot.action_features
    for key, value in action_features.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal action features: {len(action_features)}")
    
    # Get current observation using standard LeRobot interface
    print("\n" + "="*60)
    print("GETTING ROBOT OBSERVATION")
    print("="*60)
    print("Recording current robot state...")
    try:
        current_obs = robot.get_observation()
        print("✓ Successfully got robot observation")
    except Exception as e:
        print(f"❌ Failed to get robot observation: {e}")
        robot.disconnect()
        return
    
    # Extract current end-effector position from observation
    current_x = current_obs["end_effector.position.x"]
    current_y = current_obs["end_effector.position.y"]
    current_z = current_obs["end_effector.position.z"]
    current_qx = current_obs["end_effector.orientation.x"]
    current_qy = current_obs["end_effector.orientation.y"]
    current_qz = current_obs["end_effector.orientation.z"]
    current_qw = current_obs["end_effector.orientation.w"]
    
    print(f"Current observation keys: {list(current_obs.keys())}")
    print(f"Current end-effector position: x={current_x:.3f}, y={current_y:.3f}, z={current_z:.3f}")
    
    # Display joint states (positions, velocities, efforts)
    print("\nJoint States:")
    for joint_name in robot.config.ros2_interface.joint_names:
        pos = current_obs.get(f"{joint_name}.pos", 0.0)
        vel = current_obs.get(f"{joint_name}.vel", 0.0)
        effort = current_obs.get(f"{joint_name}.effort", 0.0)
        print(f"  {joint_name}: pos={pos:.3f}, vel={vel:.3f}, effort={effort:.3f}")
    
    # Display gripper state (if enabled)
    if robot.config.ros2_interface.gripper_enabled:
        gripper_joint_name = robot.config.ros2_interface.gripper_joint_name
        gripper_pos = current_obs.get(f"{gripper_joint_name}.pos", 0.0)
        print(f"\nGripper State:")
        print(f"  {gripper_joint_name}: pos={gripper_pos:.3f}")
    
    # Create target action (x + 0.1) with gripper open
    target_action = {
        "end_effector.position.x": current_x + 0.1,
        "end_effector.position.y": current_y,
        "end_effector.position.z": current_z,
        "end_effector.orientation.x": current_qx,
        "end_effector.orientation.y": current_qy,
        "end_effector.orientation.z": current_qz,
        "end_effector.orientation.w": current_qw,
        "gripper.position": 1.0  # Open gripper
    }
    
    # Create current action (back to original position) with gripper closed
    current_action = {
        "end_effector.position.x": current_x,
        "end_effector.position.y": current_y,
        "end_effector.position.z": current_z,
        "end_effector.orientation.x": current_qx,
        "end_effector.orientation.y": current_qy,
        "end_effector.orientation.z": current_qz,
        "end_effector.orientation.w": current_qw,
        "gripper.position": 0.0  # Close gripper
    }
    
    print(f"Target action: x={target_action['end_effector.position.x']:.3f}, y={target_action['end_effector.position.y']:.3f}, z={target_action['end_effector.position.z']:.3f}, gripper={target_action['gripper.position']:.1f}")
    print(f"Current action: x={current_action['end_effector.position.x']:.3f}, y={current_action['end_effector.position.y']:.3f}, z={current_action['end_effector.position.z']:.3f}, gripper={current_action['gripper.position']:.1f}")
    
    # Flag to track if robot is already disconnected
    robot_disconnected = False
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        nonlocal robot_disconnected
        print("\n\nShutting down...")
        if not robot_disconnected:
            robot.disconnect()
            robot_disconnected = True
            print("Disconnected successfully")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Position cycling parameters
    cycle_count = 0
    move_to_target = True
    move_duration = 3.0  # seconds to move between positions
    pause_duration = 1.0  # seconds to pause at each position
    
    print("\nStarting position cycling with gripper control...")
    print(f"Moving between positions every {move_duration} seconds")
    print("Gripper will open when moving to target position, close when returning")
    print("-" * 50)
    
    try:
        while True:
            cycle_count += 1
            
            if move_to_target:
                print(f"Cycle {cycle_count}: Moving to TARGET position (x + 0.1) with gripper OPEN")
                try:
                    sent_action = robot.send_action(target_action)
                    print(f"✓ Action sent successfully: gripper={sent_action.get('gripper.position', 'N/A')}")
                except Exception as e:
                    print(f"❌ Failed to send action: {e}")
                    break
                time.sleep(move_duration)
                move_to_target = False
            else:
                print(f"Cycle {cycle_count}: Moving to CURRENT position with gripper CLOSED")
                try:
                    sent_action = robot.send_action(current_action)
                    print(f"✓ Action sent successfully: gripper={sent_action.get('gripper.position', 'N/A')}")
                except Exception as e:
                    print(f"❌ Failed to send action: {e}")
                    break
                time.sleep(move_duration)
                move_to_target = True
            
            # Get current observation to monitor robot state
            try:
                current_obs = robot.get_observation()
                current_x = current_obs["end_effector.position.x"]
                current_y = current_obs["end_effector.position.y"]
                current_z = current_obs["end_effector.position.z"]
                print(f"Current robot position: x={current_x:.3f}, y={current_y:.3f}, z={current_z:.3f}")
                
                # Display joint states during movement
                print("Joint States:")
                for joint_name in robot.config.ros2_interface.joint_names:
                    pos = current_obs.get(f"{joint_name}.pos", 0.0)
                    vel = current_obs.get(f"{joint_name}.vel", 0.0)
                    effort = current_obs.get(f"{joint_name}.effort", 0.0)
                    print(f"  {joint_name}: pos={pos:.3f}, vel={vel:.3f}, effort={effort:.3f}")
                
                # Display gripper state during movement (if enabled)
                if robot.config.ros2_interface.gripper_enabled:
                    gripper_joint_name = robot.config.ros2_interface.gripper_joint_name
                    gripper_pos = current_obs.get(f"{gripper_joint_name}.pos", 0.0)
                    print(f"  {gripper_joint_name}: pos={gripper_pos:.3f}")
            except Exception as e:
                print(f"⚠️ Could not get current observation: {e}")
            
            # Pause at position
            print(f"Pausing at position for {pause_duration} seconds...")
            time.sleep(pause_duration)
            
    except KeyboardInterrupt:
        print("\nControl stopped by user")
    finally:
        if not robot_disconnected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("✓ Disconnected successfully")


if __name__ == "__main__":
    main()
