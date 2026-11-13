#!/usr/bin/env python3
"""LeRobot profile for Unitree G1D (recording / inference).

相机话题由 G1D 资产的 Camera=enable 变体发布（见
``robots/humanoid/Unitree/G1D/payloads/Camera/enable.usda``）：

  G1_Body  -> ``/head_camera/{rgb,depth,camera_info}``
  Dex1 (L) -> ``/left_hand_camera/{rgb,depth,camera_info}``
  Dex1 (R) -> ``/right_hand_camera/{rgb,depth,camera_info}``

要求场景同时开着 ``Sensor=sensors``（相机 prim 存在）和 ``EE_*=dex1``
（腕部相机来自 Dex1）。实测该组合下三路话题正常发布。

**注意命名**：这里用 ``head_camera`` / ``left_hand_camera`` 作为 LeRobot 的
相机 key，与 ROS 话题一致。而 sdg 采集内部用的短名是 ``head`` /
``wrist_left`` / ``wrist_right`` —— 两套名字指同样的物理相机，只是不同工具
的约定。做数据集对比时按话题名对齐，不要按 key 名。
"""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="unitree_g1d",
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
