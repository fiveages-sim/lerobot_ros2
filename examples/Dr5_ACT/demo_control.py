#!/usr/bin/env python3
"""
在线控制示例：加载训练好的 checkpoint（ACT 或 Diffusion Policy），用实时观测生成动作并发送给 ROS2 机器人。

假设你用本仓库录制了数据集（键名、相机、关节等一致），这里复用同样的
robot/camera 配置。若你的话题/关节名不同，请按需修改下面的 ROS2 配置。

用法示例：
    CUDA_VISIBLE_DEVICES=0 python examples/demo_control.py --dataset dataset/grasp_dataset_50_no_depth --train-config outputs/act_run2/checkpoints/last/pretrained_model/train_config.json --checkpoint outputs/act_run2/checkpoints/last --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
import torch

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
USE_STATE_INPUT = True


DEFAULT_DATASET = Path("/home/king/lerobot_ros2/dataset/down_dataset_30")
DEFAULT_TRAIN_CFG = Path("/home/king/lerobot_ros2/outputs/act_all_down30/checkpoints/last/pretrained_model/train_config.json")


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数：数据集/模型路径与控制选项。"""
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
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
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
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="推理时不向模型输入 observation.state（即使数据集中存在）。",
    )
    return parser.parse_args()


SANITIZED_DROP_INDICES: list[int] = []


def _sanitize_train_config(cfg_path: Path) -> Path:
    """
    draccus 解析严格，如果旧的 train_config.json 中包含当前版本未支持的字段，
    会直接报错（如 policy.drop_state_indices）。这里在本地预处理掉已知的多余字段，
    并将清理后的内容写到临时文件返回。
    """
    if not cfg_path.is_file():
        return cfg_path
    try:
        data = json.loads(cfg_path.read_text())
    except Exception:
        return cfg_path

    changed = False
    policy = data.get("policy")
    if isinstance(policy, dict):
        # 记录并移除旧字段
        if "drop_state_indices" in policy and isinstance(policy["drop_state_indices"], list):
            try:
                global SANITIZED_DROP_INDICES
                SANITIZED_DROP_INDICES = [int(i) for i in policy["drop_state_indices"]]
            except Exception:
                SANITIZED_DROP_INDICES = []
            policy.pop("drop_state_indices", None)
            changed = True
        if "drop_state_names" in policy:
            policy.pop("drop_state_names", None)
            changed = True

    if not changed:
        return cfg_path

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp, ensure_ascii=False, indent=2)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()
    return tmp_path


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
    """根据数据集元信息的顺序构建 state 向量。"""
    names = meta.features.get("observation.state", {}).get("names", [])
    return np.array([obs.get(name, 0.0) for name in names], dtype=np.float32)


def build_action_dict(meta: LeRobotDatasetMetadata, action_vec: np.ndarray) -> Dict[str, float]:
    """将动作向量按元信息顺序映射为命名字典。"""
    names = meta.features.get("action", {}).get("names", [])
    return {name: float(action_vec[i]) for i, name in enumerate(names)}


def _filter_state_features(meta: LeRobotDatasetMetadata, drop_set: set[str]) -> None:
    """从 meta 中移除指定的 state 维度，并同步裁剪 stats。"""
    if "observation.state" not in meta.features:
        return
    names = meta.features["observation.state"].get("names", [])
    keep_idx = [i for i, n in enumerate(names) if n not in drop_set]
    if not keep_idx:
        meta.features.pop("observation.state", None)
        meta.stats.pop("observation.state", None)
        return
    # 更新 names
    meta.features["observation.state"]["names"] = [names[i] for i in keep_idx]

    def _slice_stat(val):
        try:
            import torch
            if isinstance(val, torch.Tensor):
                if val.shape[-1] >= max(keep_idx) + 1:
                    return val[..., keep_idx]
                return val  # 维度不够时保持原样，避免越界
        except Exception:
            pass
        import numpy as np
        if isinstance(val, np.ndarray):
            if val.shape[-1] >= max(keep_idx) + 1:
                return val[..., keep_idx]
            return val
        if isinstance(val, list):
            if len(val) >= max(keep_idx) + 1:
                return [val[i] for i in keep_idx]
            return val
        return val

    stats = meta.stats.get("observation.state")
    if isinstance(stats, dict):
        for k, v in list(stats.items()):
            stats[k] = _slice_stat(v)


def init_inference(args) -> tuple[TrainPipelineConfig, LeRobotDatasetMetadata, torch.nn.Module, list[str], int | None]:
    """加载训练配置/数据集元信息/策略模型，用于在线推理。"""
    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    sanitized_cfg_path = _sanitize_train_config(args.train_config)
    train_cfg = TrainPipelineConfig.from_pretrained(sanitized_cfg_path)
    train_cfg.dataset.root = str(dataset_root)
    train_cfg.dataset.repo_id = dataset_root.name
    if args.tolerance_s is not None:
        train_cfg.dataset.tolerance_s = args.tolerance_s

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if (ckpt_path / "pretrained_model").is_dir():
            ckpt_path = ckpt_path / "pretrained_model"
    else:
        ckpt_path = args.train_config
        if ckpt_path.is_file():
            ckpt_path = ckpt_path.parent
        if ckpt_path.name != "pretrained_model":
            last_dir = ckpt_path.parent / "last" / "pretrained_model"
            if last_dir.exists():
                ckpt_path = last_dir

    train_cfg.policy.pretrained_path = str(ckpt_path)

    meta = LeRobotDatasetMetadata(train_cfg.dataset.repo_id, root=train_cfg.dataset.root)

    if train_cfg.dataset.use_imagenet_stats:
        IMAGENET_STATS = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        }
        for cam_key in meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                meta.stats[cam_key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        print(f"[info] Applied ImageNet normalization stats to {meta.camera_keys}")

    drop_state_arg = getattr(args, "drop_state_names", "")
    drop_set = set([s for s in drop_state_arg.split(",") if s.strip()]) if drop_state_arg else set()
    if SANITIZED_DROP_INDICES and "observation.state" in meta.features:
        names = meta.features["observation.state"].get("names", [])
        if len(SANITIZED_DROP_INDICES) < len(names):
            for idx in SANITIZED_DROP_INDICES:
                if 0 <= idx < len(names):
                    drop_set.add(names[idx])

    if not USE_STATE_INPUT or args.no_state:
        meta.features.pop("observation.state", None)
        meta.stats.pop("observation.state", None)
        if "robot_state_feature" in train_cfg.policy.__dict__:
            train_cfg.policy.__dict__["robot_state_feature"] = False
    else:
        if drop_set:
            _filter_state_features(meta, drop_set)
        if "observation.state" not in meta.features and "robot_state_feature" in train_cfg.policy.__dict__:
            train_cfg.policy.__dict__["robot_state_feature"] = False
        else:
            state_names = meta.features.get("observation.state", {}).get("names", [])
            state_dim = len(state_names)
            if hasattr(train_cfg.policy, "input_shapes") and isinstance(train_cfg.policy.input_shapes, dict):
                train_cfg.policy.input_shapes["observation.state"] = [state_dim]

    def build_policy(cfg, meta_obj):
        return make_policy(cfg.policy, ds_meta=meta_obj)

    try:
        policy = build_policy(train_cfg, meta)
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
            policy = build_policy(train_cfg, meta)
        else:
            raise

    policy_image_feats = list(policy.config.image_features.keys()) if hasattr(policy.config, "image_features") else []
    print(f"[info] Policy image features: {policy_image_feats}")
    print(f"[info] Dataset features: {list(meta.features.keys())}")

    policy.to(args.device)
    policy.eval()
    policy.reset()
    print("[info] Policy reset, action queue initialized")

    action_names = meta.features.get("action", {}).get("names", [])
    gripper_idx = action_names.index("gripper.position") if "gripper.position" in action_names else None

    return train_cfg, meta, policy, policy_image_feats, gripper_idx


def infer_action_vec(
    policy,
    meta: LeRobotDatasetMetadata,
    policy_image_feats: list[str],
    raw_obs: Dict[str, float],
    device: str,
    hz: float,
) -> np.ndarray | None:
    """执行一次策略推理并返回原始动作向量。"""
    batch = {}
    if USE_STATE_INPUT and "observation.state" in meta.features:
        state = torch.from_numpy(build_state_vector(meta, raw_obs)).unsqueeze(0)
        batch["observation.state"] = state

    for cam_key in policy_image_feats:
        cam_name = cam_key.replace("observation.images.", "")
        cam_name_with_dot = cam_name.replace("_rgb", ".rgb")
        candidates = [
            raw_obs.get(cam_name_with_dot),
            raw_obs.get(cam_name),
            raw_obs.get(cam_key),
        ]
        img = None
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, dict):
                img = cand.get("rgb") or cand.get("color") or cand.get("image")
            else:
                img = cand
            if img is not None:
                break

        if img is None:
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
            time.sleep(1.0 / hz)
            return None

        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[cam_key] = chw.unsqueeze(0)

    if (not USE_STATE_INPUT or "observation.state" not in meta.features) and "observation.state" not in batch:
        batch["observation.state"] = torch.zeros(1, 1, dtype=torch.float32)

    required_imgs = policy_image_feats or meta.camera_keys
    if required_imgs:
        missing = [k for k in required_imgs if k not in batch]
        if missing:
            print(f"[warn] Missing images: {missing}")
            time.sleep(1.0 / hz)
            return None

    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if v.dtype != torch.float32:
                v = v.float()
            if v.device.type != device:
                v = v.to(device)
            batch[k] = v

    action = policy.select_action(batch)
    action_vec = action.squeeze(0).cpu().numpy()
    return action_vec


def build_cmd_from_action_vec(
    meta: LeRobotDatasetMetadata,
    action_vec: np.ndarray,
    raw_obs: Dict[str, float],
    args,
    gripper_idx: int | None,
) -> Dict[str, float]:
    """将动作向量转换为机器人命令字典，并应用可选覆盖。"""
    cmd = build_action_dict(meta, action_vec)

    if args.relative:
        cmd["end_effector.position.x"] += raw_obs.get("end_effector.position.x", 0.0)
        cmd["end_effector.position.y"] += raw_obs.get("end_effector.position.y", 0.0)
        cmd["end_effector.position.z"] += raw_obs.get("end_effector.position.z", 0.0)

    if gripper_idx is not None:
        action_vec[gripper_idx] = np.clip(action_vec[gripper_idx], 0.0, 1.0)
        if args.fixed_gripper is not None:
            action_vec[gripper_idx] = np.clip(args.fixed_gripper, 0.0, 1.0)
        elif args.auto_close_z is not None:
            curr_z = raw_obs.get("end_effector.position.z")
            if curr_z is not None and curr_z <= args.auto_close_z:
                action_vec[gripper_idx] = 0.0
        cmd["gripper.position"] = float(action_vec[gripper_idx])

    return cmd


def _log_debug(args, action_vec, gripper_idx, curr_xyz, target_xyz, delta_xyz, distance) -> None:
    """打印动作与位姿差分调试信息（仅在 --debug 时生效）。"""
    if not args.debug:
        return
    print(f"[DEBUG] Raw action vector: {action_vec}")
    print(f"  Z-axis (idx 2): {action_vec[2]:.6f}")
    if gripper_idx is not None:
        print(f"  Gripper (idx {gripper_idx}): {action_vec[gripper_idx]:.6f}")
    if curr_xyz is not None:
        curr_x, curr_y, curr_z = curr_xyz
        target_x, target_y, target_z = target_xyz
        delta_x, delta_y, delta_z = delta_xyz
        print(f"Current XYZ: ({curr_x:.4f}, {curr_y:.4f}, {curr_z:.4f})")
        print(f"Target  XYZ: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
        print(f"Delta   XYZ: ({delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}), Distance: {distance:.4f}m")



def main() -> None:
    """在线控制主循环。"""
    args = parse_args()
    _train_cfg, meta, policy, policy_image_feats, gripper_idx = init_inference(args)

    robot = build_robot()
    robot.connect()
    print("✓ Robot connected, starting control loop...")

    period = 1.0 / args.hz
    try:
        while True:
            t0 = time.time()
            raw_obs = robot.get_observation()

            action_vec = infer_action_vec(
                policy=policy,
                meta=meta,
                policy_image_feats=policy_image_feats,
                raw_obs=raw_obs,
                device=args.device,
                hz=args.hz,
            )
            if action_vec is None:
                continue

            cmd = build_cmd_from_action_vec(meta, action_vec, raw_obs, args, gripper_idx)

            curr_x = raw_obs.get("end_effector.position.x")
            curr_y = raw_obs.get("end_effector.position.y")
            curr_z = raw_obs.get("end_effector.position.z")

            target_x = cmd.get("end_effector.position.x", 0.0)
            target_y = cmd.get("end_effector.position.y", 0.0)
            target_z = cmd.get("end_effector.position.z", 0.0)

            curr_xyz = None
            target_xyz = None
            delta_xyz = None
            distance = None
            if curr_x is not None:
                delta_x = target_x - curr_x
                delta_y = target_y - curr_y
                delta_z = target_z - curr_z
                distance = np.linalg.norm([delta_x, delta_y, delta_z])
                curr_xyz = (curr_x, curr_y, curr_z)
                target_xyz = (target_x, target_y, target_z)
                delta_xyz = (delta_x, delta_y, delta_z)

            _log_debug(args, action_vec, gripper_idx, curr_xyz, target_xyz, delta_xyz, distance)
            if args.debug:
                print(f"Gripper cmd: {cmd.get('gripper.position', 'N/A')}")
            robot.send_action(cmd)

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
