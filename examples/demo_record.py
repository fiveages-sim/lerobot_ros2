#!/usr/bin/env python3
"""
Simple ROS2 Data Recording Example

A simplified version of ROS2 data recording that focuses on the core functionality
without complex visualization or keyboard handling.

Usage:
    python simple_record_ros2.py

Requirements:
    - ROS2 environment must be sourced
    - ROS2 robot must be running and publishing joint states
    - ROS2 camera must be publishing images
    - lerobot_robot_ros2 and lerobot_camera_ros2 packages must be installed
"""

import time
import signal
import sys
import os
import numpy as np
from typing import Any

# Import LeRobot components
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

# Import our custom ROS2 plugins
from lerobot_robot_ros2 import ROS2RobotConfig, ROS2Robot, ROS2RobotInterfaceConfig, ControlType
from lerobot_camera_ros2 import ROS2CameraConfig


def main():
    """Simple ROS2 data recording example."""
    print("Simple ROS2 Data Recording")
    print("=" * 40)
    
    # Configuration
    FPS = 30
    EPISODE_TIME_SEC = 20
    NUM_EPISODES = 2
    
    # Create robot configuration
    camera_config = {
        "wrist_camera": ROS2CameraConfig(
            topic_name="/global_camera/rgb",
            node_name="lerobot_wrist_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8"
        )
    }
    
    ros2_interface_config = ROS2RobotInterfaceConfig(
        joint_states_topic="/joint_states",
        end_effector_pose_topic="/left_current_pose",
        end_effector_target_topic="/left_target",
        control_type=ControlType.CARTESIAN_POSE,
        joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
        max_linear_velocity=0.1,
        max_angular_velocity=0.5
    )
    
    robot_config = ROS2RobotConfig(
        id="ros2_robot",
        cameras=camera_config,
        ros2_interface=ros2_interface_config
    )
    
    # Initialize robot
    robot = ROS2Robot(robot_config)
    
    # Connect robot
    print("Connecting to robot...")
    try:
        robot.connect()
        print("✓ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Wait for robot to be ready
    time.sleep(2)
    
    # Test connection
    try:
        obs = robot.get_observation()
        print(f"✓ Robot ready - {len(obs)} features available")
    except Exception as e:
        print(f"❌ Robot test failed: {e}")
        robot.disconnect()
        return
    
    # Create dataset
    print("Creating dataset...")
    try:
        action_features = hw_to_dataset_features(robot.action_features, "action")#schema
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        
        # 获取当前执行路径
        current_dir = os.getcwd()
        dataset_name = f"ros2_dataset_{int(time.time())}"
        dataset_path = os.path.join(current_dir, dataset_name)
        
        print(f"Dataset will be saved to: {dataset_path}")
        
        dataset = LeRobotDataset.create(
            repo_id=dataset_path,  # 使用本地路径
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True
        )
        
        # 修改视频编码方法以使用H.265编码器并保留图片
        def _encode_temporary_episode_video_h265(self, video_key: str, episode_index: int) -> dict:
            """
            使用H.265编码器将PNG帧转换为MP4视频，并保留原始图片。
            """
            import tempfile
            import shutil
            from pathlib import Path
            from lerobot.datasets.video_utils import encode_video_frames
            
            temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
            img_dir = self._get_image_file_dir(episode_index, video_key)
            
            # 使用H.265编码器
            encode_video_frames(
                img_dir, 
                temp_path, 
                self.fps, 
                vcodec="hevc",  # 使用H.265编码器
                crf=23,  # 更高质量
                overwrite=True
            )
            
            # 保留原始图片（不删除img_dir）
            # shutil.rmtree(img_dir)  # 注释掉这行以保留图片
            return temp_path
        
        # 替换原始的视频编码方法
        dataset._encode_temporary_episode_video = _encode_temporary_episode_video_h265.__get__(dataset, LeRobotDataset)
        print("✓ Dataset created")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        robot.disconnect()
        return
    
    # Signal handler
    robot_disconnected = False
    
    def signal_handler(sig, frame):
        nonlocal robot_disconnected
        print("\nShutting down...")
        if not robot_disconnected:
            robot.disconnect()
            robot_disconnected = True
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Recording loop
    print(f"\nRecording {NUM_EPISODES} episodes...")
    
    try:
        obs = robot.get_observation()    
        origin_x = obs.get("end_effector.position.x", 0.0)
        origin_y = obs.get("end_effector.position.y", 0.0)
        origin_z = obs.get("end_effector.position.z", 0.0)
        origin_o_x = obs.get("end_effector.orientation.x", 0.0)
        origin_o_y = obs.get("end_effector.orientation.y", 0.0)
        origin_o_z = obs.get("end_effector.orientation.z", 0.0)
        origin_o_w = obs.get("end_effector.orientation.z", 0.0)
        for episode in range(NUM_EPISODES):
            print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
            
            start_time = time.time()
            frame_count = 0
            
                    
                # Generate simple action (small movement)
            
            
            while (time.time() - start_time) < EPISODE_TIME_SEC:
                try:
                    # Get observation
                    obs = robot.get_observation()
                    
                    # Generate simple action (small movement)
                
                    current_x = obs.get("end_effector.position.x", 0.0)
                    current_y = obs.get("end_effector.position.y", 0.0)
                    current_z = obs.get("end_effector.position.z", 0.0)
                    
                  
                    #print(current_x,current_y,current_z)
                    # Simple sinusoidal movement
                    t = time.time() - start_time
                    amplitude = 0.1  # 2cm
                    
                    action = {
                        "end_effector.position.x": current_x +amplitude * np.sin(t),
                        "end_effector.position.y": current_y +amplitude * np.cos(t),
                        "end_effector.position.z": current_z,
                        "end_effector.orientation.x": obs.get("end_effector.orientation.x", 0.0),
                        "end_effector.orientation.y": obs.get("end_effector.orientation.y", 0.0),
                        "end_effector.orientation.z": obs.get("end_effector.orientation.z", 0.0),
                        "end_effector.orientation.w": obs.get("end_effector.orientation.w", 1.0)
                

                        #"end_effector.orientation.x": obs.get("end_effector.orientation.x", 0.0),
                        #"end_effector.orientation.y": obs.get("end_effector.orientation.y", 0.0),
                        #"end_effector.orientation.z": obs.get("end_effector.orientation.z", 0.0),
                        #"end_effector.orientation.w": obs.get("end_effector.orientation.w", 1.0)
                        #pose.orientation.x=0.7077044621721283
                        #pose.orientation.y=0.7065080817531944
                        #pose.orientation.z=0.000848011543394921
                        #pose.orientation.w=-7.42664282289885e-05
                
                    }
                    
                    # Send action
                    sent_action = robot.send_action(action)
                    
                    # Prepare frame data according to dataset features
                    frame = {"task": "ros2_demo_task"}
                    
                    # Add observation state (all float features combined)
                    obs_state = []
                    # Joint positions
                    for joint_name in robot.config.ros2_interface.joint_names:
                        obs_state.append(obs.get(f"{joint_name}.pos", 0.0))
                    # Joint velocities
                    for joint_name in robot.config.ros2_interface.joint_names:
                        obs_state.append(obs.get(f"{joint_name}.vel", 0.0))
                    # Joint efforts
                    for joint_name in robot.config.ros2_interface.joint_names:
                        obs_state.append(obs.get(f"{joint_name}.effort", 0.0))
                    # Gripper position (if enabled)
                    if robot.config.ros2_interface.gripper_enabled:
                        gripper_joint_name = robot.config.ros2_interface.gripper_joint_name
                        obs_state.append(obs.get(f"{gripper_joint_name}.pos", 0.0))
                    # End-effector pose
                    obs_state.extend([
                        obs.get("end_effector.position.x", 0.0),
                        obs.get("end_effector.position.y", 0.0),
                        obs.get("end_effector.position.z", 0.0),
                        obs.get("end_effector.orientation.x", 0.0),
                        obs.get("end_effector.orientation.y", 0.0),
                        obs.get("end_effector.orientation.z", 0.0),
                        obs.get("end_effector.orientation.w", 1.0)
                    ])
                    frame["observation.state"] = np.array(obs_state, dtype=np.float32)
                    
                    # Add camera images
                    for cam_name in robot.config.cameras.keys():
                        if cam_name in obs:
                            frame[f"observation.images.{cam_name}"] = obs[cam_name]
                    
                    # Add action (all float features combined)
                    action_state = [
                        sent_action.get("end_effector.position.x", 0.0),
                        sent_action.get("end_effector.position.y", 0.0),
                        sent_action.get("end_effector.position.z", 0.0),
                        sent_action.get("end_effector.orientation.x", 0.0),
                        sent_action.get("end_effector.orientation.y", 0.0),
                        sent_action.get("end_effector.orientation.z", 0.0),
                        sent_action.get("end_effector.orientation.w", 1.0),
                    ]
                    if robot.config.ros2_interface.gripper_enabled:
                        action_state.append(sent_action.get("gripper.position", 0.0))
                    frame["action"] = np.array(action_state, dtype=np.float32)
                    
                    # Debug: Print frame info for first few frames
                    if frame_count < 3:
                        print(f"  Debug - Frame {frame_count}: {list(frame.keys())}")
                        print(f"  Debug - Episode buffer size before add_frame: {dataset.episode_buffer['size'] if dataset.episode_buffer else 'None'}")
                        print(f"  Debug - Frame task: {frame.get('task', 'MISSING')}")
                        print(f"  Debug - Frame observation.state shape: {frame.get('observation.state', 'MISSING').shape if 'observation.state' in frame else 'MISSING'}")
                        print(f"  Debug - Frame action shape: {frame.get('action', 'MISSING').shape if 'action' in frame else 'MISSING'}")
                    
                    dataset.add_frame(frame)
                    frame_count += 1
                    
                    # Debug: Check episode buffer after add_frame
                    if frame_count <= 3:
                        print(f"  Debug - Episode buffer size after add_frame: {dataset.episode_buffer['size'] if dataset.episode_buffer else 'None'}")
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0:
                        print(f"  {elapsed:.1f}s - {frame_count} frames")
                    
                    time.sleep(1.0 / FPS)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            # Save episode
            try:
                # Debug: Check episode buffer before saving
                print(f"  Debug - Before save_episode: buffer size = {dataset.episode_buffer['size'] if dataset.episode_buffer else 'None'}")
                print(f"  Debug - Frame count: {frame_count}")
                print(f"  Debug - Total episodes: {dataset.meta.total_episodes}")
                print(f"  Debug - Episode index: {dataset.episode_buffer['episode_index'] if dataset.episode_buffer else 'None'}")
                
                if dataset.episode_buffer and dataset.episode_buffer['size'] > 0:
                    dataset.save_episode()
                    print(f"✓ Episode {episode + 1} saved ({frame_count} frames)")
                else:
                    print(f"⚠️ No frames to save for episode {episode + 1}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Reset between episodes
            if episode < NUM_EPISODES - 1:
                print("Resetting...")
                reset_action = {
                    "end_effector.position.x": origin_x,
                    "end_effector.position.y": origin_y,
                    "end_effector.position.z": origin_z,
                    "end_effector.orientation.x": origin_o_x,
                    "end_effector.orientation.y": origin_o_y,
                    "end_effector.orientation.z": origin_o_z,
                    "end_effector.orientation.w": origin_o_w
                }
                robot.send_action(reset_action)
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Cleanup
        if not robot_disconnected:
            robot.disconnect()
            print("✓ Robot disconnected")
        
        try:
            # 对于本地数据集，数据已经保存在指定路径中
            print(f"✓ Dataset saved to: {dataset_path}")
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
        
        print("Recording completed!")


if __name__ == "__main__":
    main()
