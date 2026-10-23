#!/usr/bin/env python3
"""LeRobot profile for Agibot G1 (recording / inference)."""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="agibot_g1",
    cameras={
        "head": CameraTopicConfig(topic_name="/head_camera/rgb", node_name="lerobot_head_camera"),
        "left": CameraTopicConfig(topic_name="/left_camera/rgb", node_name="lerobot_left_camera"),
        "right": CameraTopicConfig(topic_name="/right_camera/rgb", node_name="lerobot_right_camera"),
    },
    depth_camera_name="",
    depth_info_topic="",
)
