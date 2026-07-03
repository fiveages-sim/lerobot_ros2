#!/usr/bin/env python3
"""LeRobot profile for Realman RM75 (recording / inference)."""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="ros2_grasp_robot",
    cameras={
        "global": CameraTopicConfig(
            topic_name="/global_camera/rgb",
            node_name="lerobot_global_camera",
            depth_topic_name="/global_camera/depth",
        ),
    },
    depth_camera_name="global",
    depth_info_topic="/global_camera/camera_info",
)
