#!/usr/bin/env python3
"""LeRobot profile for Unitree G1D (recording / online inference).

Camera keys match the FaSim collector / ACT dataset feature stems
(``observation.images.base_0_rgb``, ``left_wrist_0_rgb``, ``right_wrist_0_rgb``):
ROS2Camera async_read returns ``{rgb: ...}``, so obs keys become ``<name>.rgb``,
which ``online_infer`` maps from ``*_rgb`` policy features.
"""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="unitree_g1d",
    cameras={
        # Policy feature observation.images.base_0_rgb ← /head_camera/rgb
        "base_0": CameraTopicConfig(
            topic_name="/head_camera/rgb",
            node_name="lerobot_g1d_base_0",
        ),
        # observation.images.left_wrist_0_rgb
        "left_wrist_0": CameraTopicConfig(
            topic_name="/left_hand_camera/rgb",
            node_name="lerobot_g1d_left_wrist_0",
        ),
        # observation.images.right_wrist_0_rgb
        "right_wrist_0": CameraTopicConfig(
            topic_name="/right_hand_camera/rgb",
            node_name="lerobot_g1d_right_wrist_0",
        ),
    },
)
