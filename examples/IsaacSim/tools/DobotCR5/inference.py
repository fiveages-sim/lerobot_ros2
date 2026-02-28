#!/usr/bin/env python3
"""
在线控制示例：加载训练好的 checkpoint（ACT 或 Diffusion Policy），用实时观测生成动作并发送给 ROS2 机器人。

假设你用本仓库录制了数据集（键名、相机、关节等一致），这里复用同样的
robot/camera 配置。若你的话题/关节名不同，请按需修改下面的 ROS2 配置。

用法示例：
    python "examples/Cr5_ACT/Simulation Inference.py" \
  --dataset dataset/grasp_raw_1769487425_lerobot_v2 \
  --train-config outputs/2026-01-27/08-26-17_act/checkpoints/last/pretrained_model/train_config.json \
  --checkpoint outputs/2026-01-27/08-26-17_act/checkpoints/last \
  --device cuda \
  --hz 30

"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Callable, Dict

import numpy as np
import torch
from ros2_robot_interface import FSM_HOLD, FSM_OCS2

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot_robot_ros2 import (
    ROS2Robot,
    ROS2RobotConfig,
)

COMMON_ISAAC_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_ISAAC_DIR) not in sys.path:
    sys.path.append(str(COMMON_ISAAC_DIR))
ROBOT_CFG_DIR = Path(__file__).resolve().parents[2] / "robots" / "DobotCR5"
if str(ROBOT_CFG_DIR) not in sys.path:
    sys.path.append(str(ROBOT_CFG_DIR))

from isaac_ros2_sim_common import SimTimeHelper, reset_simulation_and_randomize_object  # pyright: ignore[reportMissingImports]
from lerobot_robot_ros2.utils.pose_utils import (  # pyright: ignore[reportMissingImports]
    quat_xyzw_to_rot6d,
    rot6d_to_quat_xyzw,
)
from flow_configs.pick_place_flow import FLOW_CONFIG as PICK_PLACE_FLOW_FLOW_CONFIG  # pyright: ignore[reportMissingImports]
from robot_config import ROBOT_CFG  # pyright: ignore[reportMissingImports]


# 全局配置
FPS = 30
# 是否向策略提供 observation.state（vision-only模型设为False）
USE_STATE_INPUT = True


DEFAULT_DATASET = None
DEFAULT_TRAIN_CFG = None
PICK_PLACE_FLOW_OVERRIDES = PICK_PLACE_FLOW_FLOW_CONFIG["base_task_overrides"]


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数：数据集/模型路径与控制选项。"""
    parser = argparse.ArgumentParser(description="Online control with ROS2 robot (ACT or Diffusion Policy).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Path to dataset root (contains meta/data/videos). "
            "If omitted, auto-detect latest dataset under ./dataset or ./lerobot_dataset."
        ),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CFG,
        help=(
            "Path to train_config.json or its parent dir (e.g. .../pretrained_model). "
            "If omitted, auto-detect latest train_config.json under ./outputs."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional explicit checkpoint dir (contains pretrained_model), e.g. .../last or .../000500. "
            "If omitted, infer from --train-config."
        ),
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


def _find_latest_dataset_dir() -> Path:
    candidates: list[Path] = []
    for parent in (Path.cwd() / "dataset", Path.cwd() / "lerobot_dataset"):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir() and (child / "meta" / "info.json").is_file():
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            "Cannot auto-detect dataset. Please pass --dataset, "
            "or create ./dataset/<name>/meta/info.json (or ./lerobot_dataset/<name>/meta/info.json)."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    print(f"[info] Auto-selected dataset: {latest}")
    return latest


def _resolve_dataset_path(dataset_arg: Path | None) -> Path:
    if dataset_arg is None:
        return _find_latest_dataset_dir()
    dataset_root = dataset_arg.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    return dataset_root


def _find_latest_train_config() -> Path:
    outputs_dir = Path.cwd() / "outputs"
    if not outputs_dir.is_dir():
        raise FileNotFoundError("Cannot auto-detect train config: ./outputs does not exist. Please pass --train-config.")

    candidates = list(outputs_dir.glob("**/checkpoints/last/pretrained_model/train_config.json"))
    if not candidates:
        candidates = list(outputs_dir.glob("**/pretrained_model/train_config.json"))
    if not candidates:
        raise FileNotFoundError(
            "Cannot auto-detect train config under ./outputs. Please pass --train-config explicitly."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    print(f"[info] Auto-selected train config: {latest}")
    return latest


def _resolve_train_config_path(train_config_arg: Path | None) -> Path:
    cfg_path = _find_latest_train_config() if train_config_arg is None else train_config_arg.expanduser().resolve()
    if cfg_path.is_dir():
        cfg_path = cfg_path / "train_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return cfg_path


def _resolve_checkpoint_path(checkpoint_arg: Path | None, train_config_path: Path) -> Path:
    if checkpoint_arg is not None:
        ckpt_path = checkpoint_arg.expanduser().resolve()
        if ckpt_path.name != "pretrained_model" and (ckpt_path / "pretrained_model").is_dir():
            ckpt_path = ckpt_path / "pretrained_model"
        if not ckpt_path.is_dir():
            raise FileNotFoundError(ckpt_path)
        return ckpt_path

    # Infer from train_config.json location first.
    if train_config_path.parent.name == "pretrained_model":
        inferred = train_config_path.parent
    else:
        inferred = train_config_path.parent / "pretrained_model"

    if inferred.is_dir():
        print(f"[info] Auto-selected checkpoint: {inferred.resolve()}")
        return inferred.resolve()

    # Fallback: try sibling last/pretrained_model from config dir.
    fallback = train_config_path.parent.parent / "last" / "pretrained_model"
    if fallback.is_dir():
        print(f"[info] Auto-selected checkpoint: {fallback.resolve()}")
        return fallback.resolve()

    raise FileNotFoundError(
        f"Cannot infer checkpoint from train config: {train_config_path}. "
        "Please pass --checkpoint explicitly."
    )


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
        if "drop_state_indices" in policy:
            try:
                global SANITIZED_DROP_INDICES
                if isinstance(policy.get("drop_state_indices"), list):
                    SANITIZED_DROP_INDICES = [int(i) for i in policy["drop_state_indices"]]
                else:
                    SANITIZED_DROP_INDICES = []
            except Exception:
                SANITIZED_DROP_INDICES = []
            policy.pop("drop_state_indices", None)
            changed = True
        if "drop_state_names" in policy:
            policy.pop("drop_state_names", None)
            changed = True

    dataset = data.get("dataset")
    if isinstance(dataset, dict):
        if "tolerance_s" in dataset:
            dataset.pop("tolerance_s", None)
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
    camera_config = {
        name: replace(cfg, fps=FPS)
        for name, cfg in ROBOT_CFG.cameras.items()
    }
    ros2_interface_config = ROBOT_CFG.ros2_interface
    robot_config = ROS2RobotConfig(
        id=f"{ROBOT_CFG.robot_id}_inference",
        cameras=camera_config,
        ros2_interface=ros2_interface_config,
        gripper_control_mode=ROBOT_CFG.gripper_control_mode,
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


def _maybe_inject_rot6d_in_obs(raw_obs: Dict[str, float], meta: LeRobotDatasetMetadata) -> None:
    names = meta.features.get("observation.state", {}).get("names", [])
    if not any(n.startswith("left_ee.rot6d_") for n in names):
        return
    quat_vals = [
        raw_obs.get("left_ee.quat.x"),
        raw_obs.get("left_ee.quat.y"),
        raw_obs.get("left_ee.quat.z"),
        raw_obs.get("left_ee.quat.w"),
    ]
    if any(v is None for v in quat_vals):
        return
    quat = np.asarray(quat_vals, dtype=np.float32)
    if np.any(np.isnan(quat)):
        return
    rot6d = quat_xyzw_to_rot6d(quat)[0]
    for i in range(6):
        raw_obs[f"left_ee.rot6d_{i}"] = float(rot6d[i])


def _maybe_convert_rot6d_action(cmd: Dict[str, float]) -> None:
    rot6d_keys = [f"left_ee.rot6d_{i}" for i in range(6)]
    if not all(k in cmd for k in rot6d_keys):
        return
    rot6d = np.array([cmd[k] for k in rot6d_keys], dtype=np.float32)
    quat_xyzw = rot6d_to_quat_xyzw(rot6d)[0]
    cmd["left_ee.quat.x"] = float(quat_xyzw[0])
    cmd["left_ee.quat.y"] = float(quat_xyzw[1])
    cmd["left_ee.quat.z"] = float(quat_xyzw[2])
    cmd["left_ee.quat.w"] = float(quat_xyzw[3])
    for k in rot6d_keys:
        cmd.pop(k, None)


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
    dataset_root = _resolve_dataset_path(args.dataset)
    train_config_path = _resolve_train_config_path(args.train_config)

    sanitized_cfg_path = _sanitize_train_config(train_config_path)
    train_cfg = TrainPipelineConfig.from_pretrained(sanitized_cfg_path)
    train_cfg.dataset.root = str(dataset_root)
    train_cfg.dataset.repo_id = dataset_root.name
    if args.tolerance_s is not None:
        train_cfg.dataset.tolerance_s = args.tolerance_s

    ckpt_path = _resolve_checkpoint_path(args.checkpoint, train_config_path)

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
    gripper_idx = action_names.index("left_gripper.pos") if "left_gripper.pos" in action_names else None

    return train_cfg, meta, policy, policy_image_feats, gripper_idx


def infer_action_vec(
    policy,
    meta: LeRobotDatasetMetadata,
    policy_image_feats: list[str],
    raw_obs: Dict[str, float],
    device: str,
    hz: float,
    sleep_fn: Callable[[float], None] = time.sleep,
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
            sleep_fn(1.0 / hz)
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
            sleep_fn(1.0 / hz)
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
        cmd["left_ee.pos.x"] += raw_obs.get("left_ee.pos.x", 0.0)
        cmd["left_ee.pos.y"] += raw_obs.get("left_ee.pos.y", 0.0)
        cmd["left_ee.pos.z"] += raw_obs.get("left_ee.pos.z", 0.0)

    if gripper_idx is not None:
        action_vec[gripper_idx] = np.clip(action_vec[gripper_idx], 0.0, 1.0)
        if args.fixed_gripper is not None:
            action_vec[gripper_idx] = np.clip(args.fixed_gripper, 0.0, 1.0)
        elif args.auto_close_z is not None:
            curr_z = raw_obs.get("left_ee.pos.z")
            if curr_z is not None and curr_z <= args.auto_close_z:
                action_vec[gripper_idx] = 0.0
        cmd["left_gripper.pos"] = float(action_vec[gripper_idx])

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
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        print("\nInterrupt received, shutting down...")
        try:
            if robot_connected:
                try:
                    robot.ros2_interface.send_fsm_command(FSM_HOLD)
                except Exception:
                    pass
                robot.disconnect()
        finally:
            sim_time.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    print("Connecting to robot...")
    robot.connect()
    robot_connected = True
    print("[OK] Robot connected")

    print("Resetting simulation state...")
    reset_simulation_and_randomize_object(
            PICK_PLACE_FLOW_OVERRIDES["source_object_entity_path"],
            xyz_offset=PICK_PLACE_FLOW_OVERRIDES["object_xyz_random_offset"],
        post_reset_wait=ROBOT_CFG.post_reset_wait,
        sleep_fn=sim_time.sleep,
    )

    print("Switching FSM: HOLD -> OCS2 ...")
    robot.ros2_interface.send_fsm_command(FSM_HOLD)
    sim_time.sleep(ROBOT_CFG.fsm_switch_delay)
    robot.ros2_interface.send_fsm_command(FSM_OCS2)
    print("[OK] FSM switched to OCS2, starting inference loop...")

    period = 1.0 / args.hz
    try:
        missing_warned = False
        while True:
            t0 = sim_time.now_seconds()
            raw_obs = robot.get_observation()
            _maybe_inject_rot6d_in_obs(raw_obs, meta)
            if not missing_warned and USE_STATE_INPUT and "observation.state" in meta.features:
                expected = meta.features.get("observation.state", {}).get("names", [])
                missing = [name for name in expected if name not in raw_obs]
                if missing:
                    print(f"[WARN] Missing observation.state keys: {missing}")
                missing_warned = True

            action_vec = infer_action_vec(
                policy=policy,
                meta=meta,
                policy_image_feats=policy_image_feats,
                raw_obs=raw_obs,
                device=args.device,
                hz=args.hz,
                sleep_fn=sim_time.sleep,
            )
            if action_vec is None:
                continue

            cmd = build_cmd_from_action_vec(meta, action_vec, raw_obs, args, gripper_idx)
            _maybe_convert_rot6d_action(cmd)

            curr_x = raw_obs.get("left_ee.pos.x")
            curr_y = raw_obs.get("left_ee.pos.y")
            curr_z = raw_obs.get("left_ee.pos.z")

            target_x = cmd.get("left_ee.pos.x", 0.0)
            target_y = cmd.get("left_ee.pos.y", 0.0)
            target_z = cmd.get("left_ee.pos.z", 0.0)

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
                print(f"Gripper cmd: {cmd.get('left_gripper.pos', 'N/A')}")
            robot.send_action(cmd)

            elapsed = sim_time.now_seconds() - t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                sim_time.sleep(sleep_t)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            if robot_connected:
                print("Switching FSM to HOLD...")
                robot.ros2_interface.send_fsm_command(FSM_HOLD)
                print("[OK] FSM switched to HOLD")
        except Exception as err:
            print(f"[WARN] Failed to switch FSM to HOLD during cleanup: {err}")
        finally:
            sim_time.shutdown()
            if robot_connected:
                robot.disconnect()
                print("✓ Robot disconnected.")


if __name__ == "__main__":
    main()
