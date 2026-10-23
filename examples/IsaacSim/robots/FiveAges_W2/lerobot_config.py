#!/usr/bin/env python3
"""LeRobot profile for FiveAges W2 (recording / inference).

Head RGB-D aligned with PhysX scene ``env/dexhand_o7_new.usda``:
``/head_camera/{rgb,depth,camera_info}`` (depth 32FC1 meters).
"""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="fiveages_w2",
    cameras={
        "head_camera": CameraTopicConfig(
            topic_name="/head_camera/rgb",
            node_name="lerobot_head_camera",
            depth_topic_name="/head_camera/depth",
        ),
        "left_hand_camera": CameraTopicConfig(
            topic_name="/left_hand_camera/rgb",
            node_name="lerobot_left_hand_camera",
        ),
        "right_hand_camera": CameraTopicConfig(
            topic_name="/right_hand_camera/rgb",
            node_name="lerobot_right_hand_camera",
        ),
    },
    depth_camera_name="head_camera",
    depth_info_topic="/head_camera/camera_info",
)
