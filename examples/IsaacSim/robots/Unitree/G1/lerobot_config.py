#!/usr/bin/env python3
"""LeRobot profile for Unitree G1 (recording / inference).

与 G1D 的 profile 完全相同的话题命名，因为两者共用同一套组件资产：
G1D 的 G1_Body / G1_Arms / Dex1 就是 G1 的那一份。

  G1_Body  -> ``/head_camera/{rgb,depth,camera_info}``
  Dex1 (L) -> ``/left_hand_camera/{rgb,depth,camera_info}``
  Dex1 (R) -> ``/right_hand_camera/{rgb,depth,camera_info}``

要求场景同时开着 ``Sensor=sensors``（相机 prim 存在）和 ``EE_*=dex1``
（腕部相机来自 Dex1）。

**注意命名**：这里用 ``head_camera`` / ``left_hand_camera`` 作为 LeRobot 的
相机 key，与 ROS 话题一致。而 sdg 采集内部用的短名是 ``head`` /
``wrist_left`` / ``wrist_right`` —— 两套名字指同样的物理相机，只是不同工具
的约定。做数据集对比时按话题名对齐，不要按 key 名。
"""

from __future__ import annotations

from robot_action_composer.config.robot_profiles import CameraTopicConfig, LeRobotRobotConfig

LEROBOT_CFG = LeRobotRobotConfig(
    robot_id="unitree_g1",
    # 注意：**CameraTopicConfig 不支持 width/height**（它只是话题描述，
    # 相机侧也不做 resize —— 只会把配置值改成实际值并告警
    # "Image dimensions mismatch: configured WxH, actual WxH"）。
    #
    # head 相机的话题分辨率由资产侧决定（当前 640x480，训练数据同分辨率）。
    # 若哪天改回 1280x720，尺寸对齐必须放在推理侧按数据集 feature shape 做
    # cv2.resize（见 online_infer/core.py），否则垂直视野差 12.3°，策略读不懂。
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
