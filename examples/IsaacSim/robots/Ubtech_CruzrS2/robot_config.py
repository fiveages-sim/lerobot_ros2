#!/usr/bin/env python3
"""Robot profile config for Ubtech Cruzr S2 IsaacSim demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot_camera_ros2 import ROS2CameraConfig  # pyright: ignore[reportMissingImports]
from ros2_robot_interface import ROS2RobotInterfaceConfig  # pyright: ignore[reportMissingImports]


ROBOT_KEY = "ubtech_cruzr_s2"
ROBOT_LABEL = "Ubtech Cruzr S2"


@dataclass(frozen=True)
class RobotConfig:
    # 夹爪控制模式：必须与 ros2_robot_interface 支持的接口一致；夹爪不响应时优先核对该项
    gripper_control_mode: str = "target_command"
    # 判定“到位”的位置阈值（米）：更小更精确但更容易超时；更大更容易跑通但精度下降
    pose_position_threshold: float = 0.05
    # 判定“到位”的姿态阈值（通常是角度误差阈值，量级为弧度）：抓取/交接对姿态敏感可调小
    pose_orientation_threshold: float = 0.1
    # ROS2 机器人接口配置：默认会在 __post_init__ 里按双臂关节名生成一份默认配置
    ros2_interface: ROS2RobotInterfaceConfig = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ros2_interface is None:
            object.__setattr__(self, "ros2_interface", ROS2RobotInterfaceConfig.default_bimanual(
                # 关节名顺序：需要与 ROS2 控制器/机器人描述里的一致，否则控制会错位或不动
                joint_names=(
                    "body_joint1",
                    "body_joint2",
                    "body_joint3",
                    "body_joint4",
                    "head_joint1",
                    "head_joint2",
                    "left_gripper_joint",
                    "left_joint1",
                    "left_joint2",
                    "left_joint3",
                    "left_joint4",
                    "left_joint5",
                    "left_joint6",
                    "left_joint7",
                    "right_gripper_joint",
                    "right_joint1",
                    "right_joint2",
                    "right_joint3",
                    "right_joint4",
                    "right_joint5",
                    "right_joint6",
                    "right_joint7",
                ),
                pose_position_threshold=self.pose_position_threshold,
                pose_orientation_threshold=self.pose_orientation_threshold,
            ))
    # Isaac Sim 中 base_link 的 Prim 路径：用于坐标/TF 对齐；路径不对会导致坐标与控制整体偏移
    base_link_entity_path: str = "/World/Ubtech_CruzrS2/Ubtech_CruzrS2_Chassis/base_link"
    # 状态机切换延迟（秒）：接口/仿真有延迟可适当调大；过小可能引起抖动或状态未稳定
    fsm_switch_delay: float = 0.1
    # reset 后等待（秒）：让仿真/TF/关节状态稳定；reset 后立即动作易导致规划失败
    post_reset_wait: float = 1.0
    # 等待到位的超时（秒）：动作幅度大/速度慢需调大；想快速失败重试可调小
    arrival_timeout: float = 5.0
    # 到位轮询间隔（秒）：更小更灵敏但占用更高；一般 0.02~0.1 合理
    arrival_poll: float = 0.05
    # 夹爪动作后额外等待（秒）：夹爪开合有机械/控制延迟时需要；过大会拖慢节奏
    gripper_action_wait: float = 0.3
    # 相机配置：key 是相机名字；ROS2CameraConfig 里填 topic_name/node_name（需与你的 ROS2 topic 实际一致）
    cameras: dict[str, ROS2CameraConfig] = field(
        default_factory=lambda: {
            # "head": ROS2CameraConfig(
            #     topic_name="/head_camera/rgb",
            #     node_name="lerobot_head_camera"
            # ),
            # "left": ROS2CameraConfig(
            #     topic_name="/left_camera/rgb",
            #     node_name="lerobot_left_camera",
            # ),
            # "right": ROS2CameraConfig(
            #     topic_name="/right_camera/rgb",
            #     node_name="lerobot_right_camera",
            # ),
        }
    )


ROBOT_CFG = RobotConfig()
