#!/usr/bin/env python3
"""LeRobot profile for FiveAges W2 (recording / inference)."""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="fiveages_w2",
    cameras={
        "head_camera": CameraTopicConfig(topic_name="/head_camera/rgb", node_name="lerobot_head_camera"),
        "left_hand_camera": CameraTopicConfig(topic_name="/left_hand_camera/rgb", node_name="lerobot_left_hand_camera"),
        "right_hand_camera": CameraTopicConfig(topic_name="/right_hand_camera/rgb", node_name="lerobot_right_hand_camera"),
    },
    depth_camera_name="",
    depth_info_topic="",
)
