#!/usr/bin/env python3
"""
在线控制示例：加载训练好的 checkpoint（ACT 或 Diffusion Policy），用实时观测生成动作并发送给 ROS2 机器人。

假设你用本仓库录制了数据集（键名、相机、关节等一致），这里复用同样的
robot/camera 配置。若你的话题/关节名不同，请按需修改下面的 ROS2 配置。

用法示例：
    CUDA_VISIBLE_DEVICES=0 /home/king/miniconda3/envs/lerobot_ros2_act/bin/python examples/demo_control.py --dataset /home/king/lerobot_ros2/dataset/grasp_dataset_50_no_depth --train-config /home/king/lerobot_ros2/outputs/act_run2/checkpoints/last/pretrained_model/train_config.json --checkpoint /home/king/lerobot_ros2/outputs/act_run2/checkpoints/last --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# 允许使用本地 vendored 的 lerobot
REPO_ROOT = Path(__file__).resolve().parents[1]
_lerobot_candidates = [
    REPO_ROOT / "lerobot" / "src",
    REPO_ROOT.parent / "lerobot" / "src",
]
for _p in _lerobot_candidates:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break
# 让本地 ROS2 插件可导入（未安装情况下）
for _p in [
    REPO_ROOT / "lerobot_robot_ros2",
    REPO_ROOT / "lerobot_camera_ros2",
]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot_robot_ros2 import (
    ControlType,
    ROS2Robot,
    ROS2RobotConfig,
    ROS2RobotInterfaceConfig,
)
from lerobot_camera_ros2 import ROS2CameraConfig


# 全局配置
FPS = 30
# 是否向策略提供 observation.state（vision-only模型设为False）
USE_STATE_INPUT = False


DEFAULT_DATASET = Path("/home/king/lerobot_ros2/dataset/down_dataset_30")
DEFAULT_TRAIN_CFG = Path("/home/king/lerobot_ros2/outputs/act_ov_down30/checkpoints/last/pretrained_model/train_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online control with ROS2 robot (ACT or Diffusion Policy).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to dataset root (contains meta/data/videos). Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CFG,
        help=f"Path to train_config.json 或其所在目录. Default: {DEFAULT_TRAIN_CFG}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="/home/king/lerobot_ros2/outputs/act_run2/checkpoints/last",
        help="可选：显式指定 checkpoint 目录（包含 pretrained_model），如 .../last 或 .../000500。",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu|mps")
    parser.add_argument("--tolerance-s", type=float, default=0.5, help="覆盖时间容差（秒），默认按 fps 计算")
    parser.add_argument("--hz", type=float, default=30.0, help="控制频率")
    parser.add_argument(
        "--relative",
        action="store_true",
        help="将模型输出的位姿当作增量，加到当前末端位姿上（仅 position）。默认按绝对坐标发送。",
    )
    parser.add_argument(
        "--fixed-gripper",
        type=float,
        default=None,
        help="可选：固定夹爪指令(0~1)，覆盖模型输出。",
    )
    parser.add_argument(
        "--auto-close-z",
        type=float,
        default=None,
        help="可选：当当前末端 z 低于该阈值(米)时自动将夹爪置为 0（仅在未设置 fixed-gripper 时生效）。",
    )
    return parser.parse_args()


def build_robot() -> ROS2Robot:
    """
    按 grasp_record.py 的设置构建 ROS2 机器人配置，确保话题/相机/关节名一致。
    """
    CAMERA_NAME = "zed"  # 改为与新数据集一致
    camera_config = {
        CAMERA_NAME: ROS2CameraConfig(
            topic_name="/global_camera/rgb",  # 话题未变
            node_name="lerobot_zed_camera",
            width=1280,
            height=720,
            fps=FPS,
            encoding="bgr8",
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
        max_angular_velocity=0.5,
        gripper_enabled=True,
        gripper_joint_name="gripper_joint",
        gripper_min_position=0.0,
        gripper_max_position=1.0,
        gripper_command_topic="gripper_joint/position_command",
    )
    robot_config = ROS2RobotConfig(
        id="ros2_grasp_robot",
        cameras=camera_config,
        ros2_interface=ros2_interface_config,
    )
    return ROS2Robot(robot_config)


def build_state_vector(meta: LeRobotDatasetMetadata, obs: Dict[str, float]) -> np.ndarray:
    names = meta.features.get("observation.state", {}).get("names", [])
    return np.array([obs.get(name, 0.0) for name in names], dtype=np.float32)


def build_action_dict(meta: LeRobotDatasetMetadata, action_vec: np.ndarray) -> Dict[str, float]:
    names = meta.features.get("action", {}).get("names", [])
    return {name: float(action_vec[i]) for i, name in enumerate(names)}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    # 加载训练配置
    train_cfg = TrainPipelineConfig.from_pretrained(args.train_config)
    train_cfg.dataset.root = str(dataset_root)
    train_cfg.dataset.repo_id = dataset_root.name
    if args.tolerance_s is not None:
        train_cfg.dataset.tolerance_s = args.tolerance_s

    # 设置权重路径：优先使用显式 checkpoint，其次使用 train_config 所在的 pretrained_model 目录。
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if (ckpt_path / "pretrained_model").is_dir():
            ckpt_path = ckpt_path / "pretrained_model"
    else:
        # 如果传入的是 train_config.json，使用其父目录；若是目录，直接用。
        ckpt_path = args.train_config
        if ckpt_path.is_file():
            ckpt_path = ckpt_path.parent
        # 若 train_config 不在 pretrained_model 内，尝试查找 last 软链
        if ckpt_path.name != "pretrained_model":
            last_dir = ckpt_path.parent / "last" / "pretrained_model"
            if last_dir.exists():
                ckpt_path = last_dir

    train_cfg.policy.pretrained_path = str(ckpt_path)

    # 仅加载元信息（含 stats 用于归一化）
    meta = LeRobotDatasetMetadata(train_cfg.dataset.repo_id, root=train_cfg.dataset.root)

    # 应用ImageNet标准化（与训练时一致）
    if train_cfg.dataset.use_imagenet_stats:
        IMAGENET_STATS = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        }
        for cam_key in meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                meta.stats[cam_key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        print(f"[info] Applied ImageNet normalization stats to {meta.camera_keys}")

    # 如果选择只用图像推理,移除状态特征以匹配无状态的模型配置
    if not USE_STATE_INPUT:
        meta.features.pop("observation.state", None)
        meta.stats.pop("observation.state", None)
        # 关闭状态 token（部分属性为只读，直接改 config 字典）
        if "robot_state_feature" in train_cfg.policy.__dict__:
            train_cfg.policy.__dict__["robot_state_feature"] = False

    # 创建策略和预后处理器，若出现 state/token 形状不匹配，自动回退为仅视觉
    def build_policy(cfg, meta_obj):
        return make_policy(cfg.policy, ds_meta=meta_obj)

    try:
        policy = build_policy(train_cfg, meta)
        state_disabled = False
    except RuntimeError as e:
        msg = str(e)
        maybe_state_mismatch = (
            "vae_encoder_pos_enc" in msg
            or "encoder_1d_feature_pos_embed" in msg
            or "size mismatch" in msg
        )
        if maybe_state_mismatch and "observation.state" in meta.features:
            print("[warn] Detected shape mismatch, retrying without observation.state ...")
            meta.features.pop("observation.state", None)
            meta.stats.pop("observation.state", None)
            if "robot_state_feature" in train_cfg.policy.__dict__:
                train_cfg.policy.__dict__["robot_state_feature"] = False
            state_disabled = True
            policy = build_policy(train_cfg, meta)
        else:
            raise

    # 获取策略实际使用的图像特征
    policy_image_feats = list(policy.config.image_features.keys()) if hasattr(policy.config, 'image_features') else []
    print(f"[info] Policy image features: {policy_image_feats}")
    print(f"[info] Dataset features: {list(meta.features.keys())}")

    # 不使用 preproc/postproc，policy.select_action 内部会处理标准化和反标准化
    policy.to(args.device)
    policy.eval()

    # 重置策略状态（初始化 action queue）
    policy.reset()
    print(f"[info] Policy reset, action queue initialized")

    robot = build_robot()
    robot.connect()
    print("✓ Robot connected, starting control loop...")

    action_names = meta.features.get("action", {}).get("names", [])
    gripper_idx = action_names.index("gripper.position") if "gripper.position" in action_names else None

    period = 1.0 / args.hz
    try:
        while True:
            t0 = time.time()
            raw_obs = robot.get_observation()

            # 组装 batch，包含 state 和相机
            batch = {}
            if USE_STATE_INPUT and "observation.state" in meta.features:
                state = torch.from_numpy(build_state_vector(meta, raw_obs)).unsqueeze(0)
                # 先保留在 CPU，预处理后统一搬到 device
                batch["observation.state"] = state
            # 加载相机图像
            for cam_key in policy_image_feats:
                # 从数据集键名提取相机名称
                # "observation.images.zed_rgb" -> "zed_rgb" -> "zed.rgb"
                # "observation.images.global_camera.rgb" -> "global_camera.rgb" -> "global_camera.rgb"
                cam_name = cam_key.replace("observation.images.", "")
                # 将 "_rgb" 转回 ".rgb" 以匹配 ROS2 观测键名
                cam_name_with_dot = cam_name.replace("_rgb", ".rgb")

                # 尝试获取图像（按优先级尝试不同键名），兼容 dict 结构
                candidates = [
                    raw_obs.get(cam_name_with_dot),  # 如 "zed.rgb"
                    raw_obs.get(cam_name),           # 如 "zed_rgb"
                    raw_obs.get(cam_key),            # 完整键名
                ]
                img = None
                for cand in candidates:
                    if cand is None:
                        continue
                    if isinstance(cand, dict):  # 例如 {"rgb":..., "depth":...}
                        img = cand.get("rgb") or cand.get("color") or cand.get("image")
                    else:
                        img = cand
                    if img is not None:
                        break
                if img is None:
                    # 模糊匹配
                    base_name = cam_name.replace("_rgb", "").replace(".rgb", "")
                    for key in raw_obs.keys():
                        if base_name in key:
                            cand = raw_obs[key]
                            if isinstance(cand, dict):
                                img = cand.get("rgb") or cand.get("color") or cand.get("image")
                            else:
                                img = cand
                            if img is not None:
                                break

                if img is None:
                    print(f"[ERROR] Could not find image for {cam_key}")
                    print(f"[DEBUG] Available obs keys: {list(raw_obs.keys())}")
                    continue

                # 处理 RGB 图像：HWC BGR/uint8 -> CHW float32 [0, 1]
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch[cam_key] = chw.unsqueeze(0)  # 先保留在 CPU

            # 如果关闭状态输入或元信息中没有 state，为模型提供占位 state（用于形状对齐）
            if (not USE_STATE_INPUT or "observation.state" not in meta.features) and "observation.state" not in batch:
                batch["observation.state"] = torch.zeros(1, 1, dtype=torch.float32)

            # 检查必需的图像是否都已加载
            required_imgs = policy_image_feats or meta.camera_keys
            if required_imgs:
                missing = [k for k in required_imgs if k not in batch]
                if missing:
                    print(f"[warn] Missing images: {missing}")
                    time.sleep(period)
                    continue

            # DEBUG: 验证图像张量是否存在
            for cam_key in policy_image_feats:
                if cam_key in batch:
                    img_tensor = batch[cam_key]
                    print(f"[DEBUG] Image tensor '{cam_key}' shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
                    print(f"[DEBUG] Image value range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
                else:
                    print(f"[ERROR] Image tensor '{cam_key}' NOT in batch!")

            # 确保所有张量都是 float32 并在正确的设备上
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    if v.dtype != torch.float32:
                        v = v.float()
                    if v.device.type != args.device:
                        v = v.to(args.device)
                    batch[k] = v

            # 直接调用 policy.select_action (内部会进行标准化)
            # 注意: 不要使用 preproc/postproc, 否则会双重标准化!
            action = policy.select_action(batch)

            # 检查 action queue 状态
            if hasattr(policy, '_action_queue'):
                print(f"[DEBUG] Action queue length: {len(policy._action_queue)}")

            # select_action 返回的是物理空间的动作，直接使用
            action_vec = action.squeeze(0).cpu().numpy()

            # 打印原始动作向量以调试
            print(f"[DEBUG] Raw action vector: {action_vec}")
            print(f"  Z-axis (idx 2): {action_vec[2]:.6f}")
            if gripper_idx is not None:
                print(f"  Gripper (idx {gripper_idx}): {action_vec[gripper_idx]:.6f}")

            # 构建动作指令
            cmd = build_action_dict(meta, action_vec)

            # 相对模式：将目标位置加到当前位置上
            if args.relative:
                cmd["end_effector.position.x"] += raw_obs.get("end_effector.position.x", 0.0)
                cmd["end_effector.position.y"] += raw_obs.get("end_effector.position.y", 0.0)
                cmd["end_effector.position.z"] += raw_obs.get("end_effector.position.z", 0.0)

            # 打印调试信息
            curr_x = raw_obs.get("end_effector.position.x")
            curr_y = raw_obs.get("end_effector.position.y")
            curr_z = raw_obs.get("end_effector.position.z")

            target_x = cmd.get("end_effector.position.x", 0)
            target_y = cmd.get("end_effector.position.y", 0)
            target_z = cmd.get("end_effector.position.z", 0)

            if curr_x is not None:
                delta_x = target_x - curr_x
                delta_y = target_y - curr_y
                delta_z = target_z - curr_z
                distance = np.linalg.norm([delta_x, delta_y, delta_z])
                print(f"Current XYZ: ({curr_x:.4f}, {curr_y:.4f}, {curr_z:.4f})")
                print(f"Target  XYZ: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
                print(f"Delta   XYZ: ({delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}), Distance: {distance:.4f}m")
            # 将末端夹抓输出限制/覆盖
            if gripper_idx is not None:
                action_vec[gripper_idx] = np.clip(action_vec[gripper_idx], 0.0, 1.0)
                if args.fixed_gripper is not None:
                    action_vec[gripper_idx] = np.clip(args.fixed_gripper, 0.0, 1.0)
                elif args.auto_close_z is not None:
                    curr_z = raw_obs.get("end_effector.position.z")
                    if curr_z is not None and curr_z <= args.auto_close_z:
                        action_vec[gripper_idx] = 0.0
                # 更新 cmd
                cmd["gripper.position"] = float(action_vec[gripper_idx])

            print(f"Gripper cmd: {cmd.get('gripper.position', 'N/A')}")

            robot.send_action(cmd)

            # 维持频率
            elapsed = time.time() - t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()
        print("✓ Robot disconnected.")


if __name__ == "__main__":
    main()
