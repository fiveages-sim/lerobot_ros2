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
from typing import Any, Callable, Dict

import numpy as np
import torch
from ros2_robot_interface import FSM_HOLD, FSM_OCS2

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot_robot_ros2 import (
    ROS2Robot,
)

from isaac_ros2_sim_common import SimTimeHelper  # pyright: ignore[reportMissingImports]
from lerobot_robot_ros2.utils.pose_utils import (  # pyright: ignore[reportMissingImports]
    quat_xyzw_to_rot6d,
    rot6d_to_quat_xyzw,
)


# 全局配置
FPS = 30
# 是否向策略提供 observation.state（vision-only模型设为False）
USE_STATE_INPUT = True


DEFAULT_DATASET = None
DEFAULT_TRAIN_CFG = None
MOTION_CFG: Any | None = None
LEROBOT_CFG: Any | None = None
ROBOT_CFG: Any | None = None  # deprecated alias for MOTION_CFG
PICK_PLACE_FLOW_OVERRIDES: dict[str, Any] = {}


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
    parser.add_argument(
        "--no-base-lock",
        dest="no_base_lock",
        action="store_true",
        help="不锁底盘（默认会发 BASE_LOCK，与采集时的编排保持一致）。"
             "只有需要 WBC 同时挪底盘时才用。",
    )
    parser.add_argument(
        "--n-action-steps",
        dest="n_action_steps",
        type=int,
        default=25,
        help="覆盖 checkpoint 的 n_action_steps：一次预测后连续执行多少帧再回读观测。"
             "checkpoint 常是 1，配 chunk_size=100 会导致闭环死锁（运控稳态残差约"
             "2.5mm > 数据每帧 1.66mm，增量被残差吃掉，目标每帧倒退）。"
             "默认 25；设 0 = 用 checkpoint 原值。",
    )
    parser.add_argument(
        "--settle-s",
        dest="settle_s",
        type=float,
        default=1.0,
        help="进入 OCS2 后、开始推理前原地保持多少秒（等 MPC 收敛、姿态稳定）。"
             "默认 1.0；设 0 = 不等待。",
    )
    parser.add_argument(
        "--clock",
        dest="clock",
        type=str,
        default="wall",
        choices=("sim", "wall"),
        help="循环节拍按哪个时钟走。"
             "sim = 仿真时间（与采集时间轴语义一致，但仿真 RTF 低时会把动作拉慢）；"
             "wall = 墙钟（保证实际吞吐 20Hz）。"
             "实测闭环只有 1.4Hz 时用 wall；训练数据是 20Hz 仿真时间，"
             "理想情况两钟一致（RTF=1）。默认 wall。",
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


def _wait_for_scene_ready(robot: ROS2Robot, timeout_s: float = 15.0) -> bool:
    """只读等待：轮询观测里的末端位姿，直到拿到非零值。

    **不写任何仿真状态** —— 复位/开关仿真由人负责。这里只确认：
      - Isaac 在跑（`/left_current_pose` 之类有数据）
      - 运控在跑（末端位姿在更新）

    为什么不用 `/get_entity_state` 等物体稳定：那属于复位流程的一部分，
    我们不再复位场景，也就不需要等物体。推理开始时物体应在人摆好的位置。
    """
    import time as _t

    keys = ("left_ee.pos.x", "right_ee.pos.x")
    t0 = _t.monotonic()
    announced = False
    while _t.monotonic() - t0 < timeout_s:
        try:
            obs = robot.get_observation()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读取观测失败: {exc}")
            obs = {}
        vals = {k: obs.get(k) for k in keys if obs.get(k) is not None}
        # 至少要有一侧末端位姿非零（(0,0,0) 说明话题还没出数据）
        if any(abs(float(v)) > 1e-6 for v in vals.values()):
            names = ", ".join(f"{k}={float(v):+.3f}" for k, v in vals.items())
            print(f"[OK] 场景就绪（{_t.monotonic() - t0:.1f}s）：{names}")
            return True
        if not announced:
            print(f"[info] 等待末端位姿话题出数据（最多 {timeout_s:.0f}s）...")
            announced = True
        _t.sleep(0.3)

    print(f"[ERROR] {timeout_s:.0f}s 内没等到末端位姿 —— 请确认：")
    print("        1) Isaac 已 Play，运控（/ocs2_wbc_controller）已启动")
    print("        2) 终端与 Isaac 使用相同的 ROS_DOMAIN_ID / zenoh 配置")
    print("        3) 手动验证：ros2 topic echo /left_current_pose --once")
    return False


def build_robot() -> ROS2Robot:
    """
    按 motion + lerobot 配置构建 ROS2 机器人，确保话题/相机/关节名与录制一致。
    """
    from robot_action_composer.config.lerobot_bridge import apply_fps_to_cameras, build_ros2_robot_config

    motion_cfg = MOTION_CFG if MOTION_CFG is not None else ROBOT_CFG
    if motion_cfg is None or LEROBOT_CFG is None:
        raise RuntimeError(
            "MOTION_CFG and LEROBOT_CFG are not configured. Run via examples/IsaacSim/inference.py."
        )
    robot_config = build_ros2_robot_config(
        motion_cfg=motion_cfg,
        lerobot_cfg=LEROBOT_CFG,
        fps=int(FPS),
        robot_id_suffix="_inference",
    )

    # ---- 关掉「发目标前自动切 FSM」----
    #
    # ArmHandler.send_target() 里有一段（handler/arm_handler.py:188）：
    #     if config.auto_switch_fsm_before_control:
    #         if query_fsm_command() != FSM_OCS2:
    #             fsm_command_callback(FSM_OCS2)      # ← 每帧都可能触发
    # 而该 callback 绑的是 send_fsm_command()，它**强制先回 HOLD**
    # （ros_interface.py:2626 `hold_required_targets` 逻辑）。
    #
    # 一旦离开 OCS2，运控会调 resetHumanoidModeToFsmDefault()
    # （Ocs2WbcController.cpp:100 `if (leaving_ocs2)`），
    # 把全身模式设回 fsm_default_humanoid_mode_ —— **BASE_LOCK 被清掉**。
    # 表现就是「推理启动时锁了底盘，跑着跑着底盘开始漂」。
    #
    # 更隐蔽的是 get_fsm_state() 读的是 /fsm_state（transient_local）缓存值，
    # 仿真刚 reset、时间戳跳变时它可能是陈旧的，于是误判"不在 OCS2"
    # 而反复触发切换，把锁住的底盘一次次踢开。
    #
    # 推理场景下 FSM 由 core.py 显式管理（HOLD→OCS2 一次），
    # 不需要 send_target 每帧自查，所以这里关掉。
    iface_cfg = getattr(robot_config, "ros2_interface", None)
    if iface_cfg is not None and getattr(iface_cfg, "auto_switch_fsm_before_control", False):
        iface_cfg.auto_switch_fsm_before_control = False
        print("[info] auto_switch_fsm_before_control=False "
              "（防止 send_target 偷偷回 HOLD 把 BASE_LOCK 清掉）")
        # 注意：ROS2RobotInterface.__init__（ros_interface.py:248）会把这个值
        # 复制成私有成员 _auto_switch_fsm_before_control，改 config 不影响它。
        # 所以 connect() 之后还必须同步私有成员（见 _enforce_no_auto_fsm_switch）。


    camera_config = apply_fps_to_cameras(robot_config.cameras, int(FPS))
    robot_config = replace(robot_config, cameras=camera_config)
    return ROS2Robot(robot_config)


def build_state_vector(meta: LeRobotDatasetMetadata, obs: Dict[str, float]) -> np.ndarray:
    """根据数据集元信息的顺序构建 state 向量。

    关节名后缀要兼容两种写法：
      - 数据集里是裸关节名（sdg 导出的 ``waist_yaw_joint``）
      - ROS2Robot 给的是 ``waist_yaw_joint.pos`` / ``.vel`` / ``.effort``

    所以按名字找不到时，依次尝试 ``<name>.pos``（最常见）、``<name>``。
    两样都没有才填 0（并计数告警，避免静默喂全零给策略）。
    """
    names = meta.features.get("observation.state", {}).get("names", [])
    if not names:
        return np.zeros(0, dtype=np.float32)

    out = np.zeros(len(names), dtype=np.float32)
    missing: list[str] = []
    for i, name in enumerate(names):
        for cand in (name, f"{name}.pos"):
            if cand in obs:
                out[i] = float(obs[cand])
                break
        else:
            missing.append(name)

    if missing and not getattr(build_state_vector, "_warned", False):
        build_state_vector._warned = True  # type: ignore[attr-defined]
        sample = list(obs.keys())[:6]
        print(f"[warn] {len(missing)}/{len(names)} 个 state 维度在观测里找不到，"
              f"已填 0（示例 {missing[:4]}）。观测里的键示例: {sample}")
    return out


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

    # ---- 关键：提高 n_action_steps ----
    #
    # checkpoint 里存的是 n_action_steps=1（训练脚本的默认值），配合
    # chunk_size=100 意味着"预测 100 帧、只执行第 0 帧、然后重新预测"。
    #
    # 实测这会导致**死锁**（见 docs/DEBUG_LOG.md 第 16 节）：
    #   运控稳态残差 r≈2.5mm（current_pose 恒落后指令目标）
    #   策略每帧增量  d≈1.66mm（训练数据真实量级）
    #   target_new = current_pose + d = target_old - 2.5 + 1.66
    #              = target_old - 0.84mm   ← 目标每帧倒退！
    # 结果 800 帧目标只净动 8.5mm，方向还被下垂噪声主导（cos=-0.96）。
    #
    # 修法：一次预测、连续执行 N 帧不回读观测。执行的是 chunk 内的**绝对序列**，
    # 不受残差反馈影响。实测单 chunk 净跨度 100mm+，足够走完 25cm 任务。
    #
    # 注意 `select_action` 内部有 action queue：队列空才重新预测。
    # 所以只需设好 config，不需要在外面自己维护队列。
    override = getattr(args, "n_action_steps", None)
    if override is not None and override > 0:
        old = policy.config.n_action_steps
        cap = min(int(override), policy.config.chunk_size)
        policy.config.n_action_steps = cap
        if cap != old:
            print(f"[info] n_action_steps: {old} → {cap} "
                  f"(chunk_size={policy.config.chunk_size})")
            print(f"[info]   ⇒ 一次预测执行 {cap} 帧再回读观测，避免稳态残差吃掉增量")
        if cap > 1 and policy.config.temporal_ensemble_coeff is not None:
            print("[warn] temporal_ensemble_coeff 非 None 时要求 n_action_steps==1；"
                  "已把 temporal ensemble 关掉")
            policy.config.temporal_ensemble_coeff = None
    elif policy.config.n_action_steps == 1 and policy.config.chunk_size > 1:
        print("[warn] n_action_steps=1 + chunk_size="
              f"{policy.config.chunk_size}：闭环里会因稳态残差**倒退**。"
              "建议加 --n-action-steps 25")

    policy.reset()
    print("[info] Policy reset, action queue initialized")

    action_names = meta.features.get("action", {}).get("names", [])

    def _find_idx(name: str) -> int | None:
        return action_names.index(name) if name in action_names else None

    # 夹爪在 action 里的位置（``ee_absolute_gripper`` 才有）。
    #
    # 语义是**开关量** 0/1，对齐运控 ``target_command``：
    #   0 = 闭合 → AdaptiveGripperController 回 ``closed_position_``（限位），
    #       靠压出夹持力夹紧物体
    #   1 = 张开
    # 所以这里**绝不能**把连续值当关节位置发 —— 那样会停在"刚好碰到管子"，
    # 夹持力不足（踩过的坑）。发送前统一 round 到 0/1。
    gripper_idx = _find_idx("left_gripper.pos")
    right_gripper_idx = _find_idx("right_gripper.pos")
    if gripper_idx is not None:
        print(f"[info] 夹爪通道: left idx={gripper_idx}"
              + (f", right idx={right_gripper_idx}" if right_gripper_idx is not None else "")
              + "（开关量 0/1 → target_command）")

    # ---- 加载 pre/post processor（归一化）----
    # 必须从 checkpoint 目录读：训练时自动保存了
    #   policy_preprocessor.json  + ..._normalizer_processor.safetensors
    #   policy_postprocessor.json + ..._unnormalizer_processor.safetensors
    # 不加载的话 select_action 会拿原始观测当归一化输入、输出当归一化值直接用。
    pre = post = None
    try:
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        ckpt_dir = Path(str(ckpt_path))
        for fname, which in (("policy_preprocessor.json", "pre"),
                             ("policy_postprocessor.json", "post")):
            if (ckpt_dir / fname).is_file():
                pipe = PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=str(ckpt_dir),
                    config_filename=fname,
                    to_transition=None,
                )
                if which == "pre":
                    pre = pipe
                else:
                    post = pipe
        if pre is None or post is None:
            print("[warn] 未能加载 pre/post processor —— 归一化缺失，"
                  "推理结果会严重偏差（动作目标会跑到错误位置）")
        else:
            print("[info] 归一化 processor 已加载")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] processor 加载失败: {exc}")

    return train_cfg, meta, policy, policy_image_feats, gripper_idx, pre, post, right_gripper_idx


def _camera_alias_map(meta: LeRobotDatasetMetadata) -> dict[str, str]:
    """数据集相机 key → ROS2Robot observation key 的映射。

    **为什么需要**：两套工具给同一台相机起了不同的名字。

    | 来源 | head | 左腕 | 右腕 |
    |---|---|---|---|
    | sdg 采集（数据集 feature key） | ``head`` | ``wrist_left`` | ``wrist_right`` |
    | lerobot_robot_ros2（话题名） | ``head_camera`` | ``left_hand_camera`` | ``right_hand_camera`` |

    ROS2Robot 的 observation key 由 ``lerobot_config.py`` 的 cameras 决定；
    有深度话题的相机拆成 ``<key>.rgb`` / ``<key>.depth``。

    这里按「包含关系 + 方位词」做启发式匹配，避免写死映射表（换机器人也能用）。
    """
    # 从 robot 侧已知 key 里找（LEROBOT_CFG 可能在 inference.py 里注入）
    robot_keys: list[str] = []
    try:
        cfg = LEROBOT_CFG
        if cfg is not None and getattr(cfg, "cameras", None):
            robot_keys = list(cfg.cameras.keys())
    except Exception:  # noqa: BLE001
        robot_keys = []

    ds_keys = [k.replace("observation.images.", "") for k in (meta.camera_keys or [])]
    if not robot_keys or not ds_keys:
        return {k: k for k in ds_keys}

    def _tokens(s: str) -> set[str]:
        return {t for t in s.replace("-", "_").split("_") if t and t not in {"camera", "rgb"}}

    alias: dict[str, str] = {}
    for dk in ds_keys:
        # 1) 完全同名
        if dk in robot_keys:
            alias[dk] = dk
            continue
        # 2) 方位词匹配（head / left / right / wrist / hand）
        d_tok = _tokens(dk)
        best, best_score = None, 0
        for rk in robot_keys:
            r_tok = _tokens(rk)
            score = len(d_tok & r_tok)
            if score > best_score:
                best, best_score = rk, score
        alias[dk] = best or dk
    return alias


def infer_action_vec(
    policy,
    meta: LeRobotDatasetMetadata,
    policy_image_feats: list[str],
    raw_obs: Dict[str, float],
    device: str,
    hz: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    preprocessor=None,
    postprocessor=None,
) -> np.ndarray | None:
    """执行一次策略推理并返回原始动作向量。

    **归一化必须由调用方通过 ``preprocessor`` / ``postprocessor`` 传入**：
    lerobot 0.5.x 的 ``policy.select_action()`` 只做模型前向，既不归一化输入
    也不反归一化输出。漏掉这一步的症状（实测）：
      - 输入喂的是原始米级坐标（x≈2.1），模型期待归一化值
      - 输出是归一化空间的值（≈0.85），被当成米发给运控 → 目标跑到
        (0.85, -0.87, -0.95)，与当前位置差 2.37m，机器人纹丝不动
    这就是「机器人不动、只有腰部微抬」的成因。
    """
    batch = {}
    if USE_STATE_INPUT and "observation.state" in meta.features:
        state = torch.from_numpy(build_state_vector(meta, raw_obs)).unsqueeze(0)
        batch["observation.state"] = state.to(device)

    # 相机 key 映射（数据集名 → ROS2Robot 名）。缓存在函数属性里，
    # 避免每帧重算。
    alias = getattr(infer_action_vec, "_alias_cache", None)
    if alias is None:
        alias = _camera_alias_map(meta)
        infer_action_vec._alias_cache = alias  # type: ignore[attr-defined]
        print(f"[info] camera alias: {alias}")

    for cam_key in policy_image_feats:
        cam_name = cam_key.replace("observation.images.", "")
        mapped = alias.get(cam_name, cam_name)
        cam_name_with_dot = mapped.replace("_rgb", ".rgb")
        candidates = [
            raw_obs.get(f"{cam_name_with_dot}.rgb"),   # 有深度的相机 → head_camera.rgb
            raw_obs.get(cam_name_with_dot),
            raw_obs.get(mapped),
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

        # ---- 尺寸对齐到训练数据（**必须**）----
        #
        # 训练数据三路都是 640x480；而 G1D 的 Camera=enable 发布的 head 是
        # 1280x720（腕部恰好是 640x480）。不 resize 的话策略看到的头部画面
        # 是"缩放过的世界" —— 场景在画面里比例/位置全变，策略读不懂，
        # 输出退化成原地微动甚至乱动（实测：手臂拧成麻花）。
        #
        # 期望尺寸从数据集 feature shape 读：observation.images.<name> =
        # [H, W, 3]。
        feat = (meta.features or {}).get(cam_key) or {}
        want_shape = feat.get("shape") or []
        if len(want_shape) >= 2:
            want_h, want_w = int(want_shape[0]), int(want_shape[1])
            if img.shape[0] != want_h or img.shape[1] != want_w:
                import cv2

                img = cv2.resize(img, (want_w, want_h), interpolation=cv2.INTER_AREA)
                if not getattr(infer_action_vec, "_resize_warned", False):
                    infer_action_vec._resize_warned = True  # type: ignore[attr-defined]
                    print(f"[info] 相机 {cam_name} 已 resize 到训练尺寸 "
                          f"{want_w}x{want_h}（原 {img.shape[1]}x{img.shape[0]}）")

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

    # ---- 归一化 → 前向 → 反归一化 ----
    # 顺序不能改。缺了这两步，模型拿原始米级坐标当归一化输入、又把归一化输出
    # 当米发出去（实测：目标 (0.85,-0.87,-0.95) vs 当前 (2.11,-0.03,0.88)，
    # 差 2.37m，机器人不动）。
    if preprocessor is not None:
        batch = preprocessor(batch)
    action = policy.select_action(batch)
    if postprocessor is not None:
        out = postprocessor({"action": action})
        action = out.get("action", action) if isinstance(out, dict) else out
    action_vec = action.squeeze(0).detach().cpu().numpy()
    return action_vec


def _binarize_gripper(value: float, threshold: float = 0.5) -> float:
    """把策略输出的夹爪值变成 0/1 开关量。

    训练时 action 里就是 0/1，但回归出来的会落在中间（比如 0.83、0.12）。
    运控的 ``target_command`` 只认 0/1，所以这里必须二值化 —— 否则
    ``_to_target_command`` 会把中间值按 0.5 阈值再判一次，行为一致但不可控。

    **不要**把这里当成位置下发：0/1 走的是 ``closed_position_`` / 
    ``open_position_``，闭合幅度由 AdaptiveGripperController 决定（回限位
    压出夹持力），不是策略给的数值。
    """
    return 0.0 if float(value) < float(threshold) else 1.0


def build_cmd_from_action_vec(
    meta: LeRobotDatasetMetadata,
    action_vec: np.ndarray,
    raw_obs: Dict[str, float],
    args,
    gripper_idx: int | None,
    right_gripper_idx: int | None = None,
) -> Dict[str, float]:
    """将动作向量转换为机器人命令字典，并应用可选覆盖。

    夹爪两维（``left_gripper.pos`` / ``right_gripper.pos``）在这一步二值化成
    0/1，对齐运控 ``target_command`` 的开关语义。
    """
    cmd = build_action_dict(meta, action_vec)

    if args.relative:
        cmd["left_ee.pos.x"] += raw_obs.get("left_ee.pos.x", 0.0)
        cmd["left_ee.pos.y"] += raw_obs.get("left_ee.pos.y", 0.0)
        cmd["left_ee.pos.z"] += raw_obs.get("left_ee.pos.z", 0.0)

    for idx, key in ((gripper_idx, "left_gripper.pos"),
                     (right_gripper_idx, "right_gripper.pos")):
        if idx is None:
            continue
        action_vec[idx] = np.clip(action_vec[idx], 0.0, 1.0)
        if args.fixed_gripper is not None:
            action_vec[idx] = np.clip(args.fixed_gripper, 0.0, 1.0)
        elif (idx == gripper_idx and args.auto_close_z is not None):
            curr_z = raw_obs.get("left_ee.pos.z")
            if curr_z is not None and curr_z <= args.auto_close_z:
                action_vec[idx] = 0.0
        cmd[key] = _binarize_gripper(action_vec[idx])

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



def bringup_control(
    robot,
    args,
    sim_time,
    *,
    motion_cfg=None,
    want_lock: bool = True,
) -> bool:
    """把运控从「待命」带到「OCS2 + 双臂使能 + 底盘锁定」的推理就绪状态。

    这段是**推理与开环回放共用**的启动序列 —— 两条路径必须走完全相同的
    运控准备，否则工况不一致会让结果不可比。抽出来就是为了避免两边漂移。

    顺序（每一步都有踩坑记录，见 docs/DEBUG_LOG.md）::

        1. HOLD                          ← 全身模式只在 HOLD 里下发才不被覆盖
        2. ARMS_INDEPENDENT + HOME_JOINT_OFF + BASE_LOCK
        3. 播种 EE 目标 = 当前实测位姿   ← 防进 OCS2 瞬间被拽向零位
        4. 切 OCS2
        5. re-apply 一遍模式             ← on_activate 会装回 fsm_default
        6. 使能双臂
        7. 原地保持 settle_s 秒          ← 等 MPC 收敛
        8. 终检（含补发）

    Returns:
        True 表示终检通过。
    """
    iface = robot.ros2_interface
    motion_cfg = motion_cfg if motion_cfg is not None else (MOTION_CFG or ROBOT_CFG)

    # ---------- 局部辅助 ----------
    def _fsm_state() -> int | None:
        getter = getattr(iface, "get_fsm_state", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:  # noqa: BLE001
            return None

    def _send_modes(cmds: list, require: bool = True, timeout: float = 5.0) -> bool:
        """下发一组 mode 命令并等运控在 current_state 上确认。"""
        for c in cmds:
            try:
                iface.send_mode_command(c)
                print(f"[OK] mode command: {c}")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {c} 发送失败: {exc}")
                if require:
                    return False
        waiter = getattr(iface, "wait_until_mode_commands_applied", None)
        if waiter is None:
            sim_time.sleep(0.6)     # 旧接口没有确认 API，退化为等一会
            return True
        try:
            ok = bool(waiter(cmds, timeout=timeout,
                             time_now_fn=sim_time.now_seconds,
                             sleep_fn=sim_time.sleep))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 等待模式确认异常: {exc}")
            ok = False
        if ok:
            print(f"[OK] modes confirmed: {cmds}")
        else:
            msg = f"modes {cmds} 在 {timeout:.0f}s 内未确认"
            print(f"[{'ERROR' if require else 'warn'}] {msg}")
            if require:
                return False
        return True

    def _ensure_fsm(target: int, name: str, timeout: float = 8.0) -> bool:
        """把 FSM 切到 target 并确认。send_fsm_command 内部会按需先回 HOLD。"""
        import time as _t
        if _fsm_state() == target:
            print(f"[OK] FSM 已在 {name}")
            return True
        print(f"[info] FSM={_fsm_state()} → {name} ...")
        robot.ros2_interface.send_fsm_command(target)
        t0 = _t.monotonic()
        while _t.monotonic() - t0 < timeout:
            if _fsm_state() == target:
                print(f"[OK] FSM={name}（已确认）")
                return True
            _t.sleep(0.1)
        print(f"[warn] {timeout:.0f}s 内未确认 FSM={name}（当前 {_fsm_state()}）")
        return False

    def _ensure_ocs2(timeout: float = 8.0) -> bool:
        return _ensure_fsm(FSM_OCS2, "OCS2", timeout=timeout)

    # ---- 1) 先回 HOLD：只有在 HOLD 里下发的全身模式才不会被 OCS2 默认值覆盖 ----
    _ensure_fsm(FSM_HOLD, "HOLD")

    # ---- 2) 在 HOLD 里把全身模式一次性设好 ----
    # body 模式不发（编排也不发，保持 fsm_default = BODY_VERTICAL）。
    boot_modes = ["ARMS_INDEPENDENT",
                  "HOME_JOINT_OFF"]      # 进 OCS2 时不往全零姿态拽
    if want_lock:
        boot_modes.append("BASE_LOCK")
    print(f"[info] 在 HOLD 里应用全身模式: {boot_modes}")
    _send_modes(boot_modes, require=False, timeout=5.0)

    # ---- 3) 播种 EE 目标 = 当前实测位姿 ----
    #
    # 进 OCS2 的瞬间若还没有任何末端目标，运控会沿用缓存/零位目标，手臂会
    # 先"掉"一下再恢复（实测约 2s 的下垂）。先把两侧目标设成当前实测位姿，
    # 进 OCS2 后手臂就原地保持。
    from geometry_msgs.msg import Pose as _Pose

    def _pose_of(obs: Dict[str, float], side: str) -> _Pose:
        p = _Pose()
        p.position.x = float(obs.get(f"{side}_ee.pos.x", 0.0))
        p.position.y = float(obs.get(f"{side}_ee.pos.y", 0.0))
        p.position.z = float(obs.get(f"{side}_ee.pos.z", 0.0))
        p.orientation.x = float(obs.get(f"{side}_ee.quat.x", 0.0))
        p.orientation.y = float(obs.get(f"{side}_ee.quat.y", 0.0))
        p.orientation.z = float(obs.get(f"{side}_ee.quat.z", 0.0))
        p.orientation.w = float(obs.get(f"{side}_ee.quat.w", 1.0))
        return p

    def _seed_ee_hold_current(obs: Dict[str, float]) -> int:
        n = 0
        for side in ("left", "right"):
            if obs.get(f"{side}_ee.pos.x") is None:
                continue
            handler = getattr(iface, f"{side}_arm_handler", None)
            if handler is None or not hasattr(handler, "send_target"):
                continue
            try:
                handler.send_target(_pose_of(obs, side))
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 播种 {side} EE 目标失败: {exc}")
        return n

    try:
        raw_now = robot.get_observation()
    except Exception as exc:  # noqa: BLE001
        raw_now = {}
        print(f"[warn] 读观测失败，跳过 EE 播种: {exc}")
    if raw_now:
        seeded = _seed_ee_hold_current(raw_now)
        print(f"[info] 已在 HOLD 播种 EE 目标（{seeded} 侧 = 当前实测位姿）")
        sim_time.sleep(0.1)

    # ---- 4) 切 OCS2（此时底盘已锁、HOME 已关、EE 目标已在实测位姿）----
    _ensure_ocs2()

    # ---- 5) 再 re-apply 一遍：on_activate 会把 fsm_default 装回去 ----
    print(f"[info] OCS2 激活后重申全身模式: {boot_modes}")
    _send_modes(boot_modes, require=False, timeout=5.0)

    # ---- 6) 使能双臂 ----
    # 编排里 LEFT_ARM_DISABLE（抓取前关左臂）是任务层的事，推理不照做：
    # 策略输出 16 维含左臂，双臂都要使能。
    _send_modes(["LEFT_ARM_ENABLE", "RIGHT_ARM_ENABLE"], require=True, timeout=5.0)

    # ---- 7) 进 OCS2 后原地保持一会儿，等姿态稳定再交策略 ----
    #
    # 进 OCS2 时 MPC 需要几个周期收敛到"原地保持"，期间手臂可能有几毫米
    # 的沉降。等它稳了再开始推理，否则策略第一帧看到的是"正在沉降的画面"。
    settle_s = float(getattr(args, "settle_s", 1.0) or 0.0)
    if settle_s > 0:
        print(f"[info] 等待 OCS2 姿态稳定 {settle_s:.1f}s ...")
        sim_time.sleep(settle_s)

    sim_time.sleep(motion_cfg.fsm_switch_delay)

    # ---- 终检：确认 FSM 仍 OCS2 且锁/使能都在 ----
    # 上面每步都确认过，但 FSM 切换是异步的；如果切换在模式下发之后才落地，
    # 前面的确认就白费了。所以最后再核一遍，不对就补（只补模式，不再切 FSM）。
    def _final_check() -> bool:
        checker = getattr(iface, "mode_commands_match_wbc_state", None)
        if checker is None:
            print("[warn] 接口没有 mode_commands_match_wbc_state，跳过终检")
            return True
        cmds = (["BASE_LOCK"] if want_lock else []) + \
               ["LEFT_ARM_ENABLE", "RIGHT_ARM_ENABLE", "HOME_JOINT_OFF"]
        for attempt in (1, 2, 3):
            fsm_left_ocs2 = False
            if _fsm_state() not in (None, FSM_OCS2):
                print(f"[warn] 终检发现 FSM 离开了 OCS2（{_fsm_state()}）—— 重新切回")
                _ensure_ocs2()
                # 切 FSM 会让运控 resetHumanoidModeToFsmDefault() 把 BASE_LOCK 清掉，
                # 所以必须无条件重发模式 —— 不能等下面 checker 发现。
                fsm_left_ocs2 = True
            try:
                ok = checker(cmds)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 终检异常: {exc}")
                ok = False if fsm_left_ocs2 else None
            if ok is True and not fsm_left_ocs2:
                print(f"[OK] 终检通过：FSM=OCS2，{cmds} 全部生效")
                return True
            print(f"[warn] 终检未通过（第 {attempt} 次），补发 {cmds} ...")
            for c in cmds:
                try:
                    iface.send_mode_command(c)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] 补发 {c} 失败: {exc}")
            sim_time.sleep(0.8)
        print("[ERROR] 终检连续失败 —— 底盘可能仍会漂移 / 手臂可能不跟随。")
        print("        建议先跑：python exts/fasim.replicator.sdg/scripts/probe_ee_tracking.py --check-only")
        return False

    return _final_check()


def main() -> None:
    """在线控制主循环。"""
    args = parse_args()
    _train_cfg, meta, policy, policy_image_feats, gripper_idx, _pre, _post, right_gripper_idx = init_inference(args)

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

    # ---- 同步「自动切 FSM」开关到接口的私有成员 ----
    #
    # ROS2RobotInterface.__init__（ros_interface.py:248）把
    # config.auto_switch_fsm_before_control 复制成 _auto_switch_fsm_before_control，
    # 而 ArmHandler.send_target 读的是后者。只改 config 不会生效，
    # 必须在这里同步；否则 send_target 仍会在"以为不在 OCS2"时发 HOLD，
    # 把 BASE_LOCK 清掉（详见 build_robot 里的注释）。
    iface0 = getattr(robot, "ros2_interface", None)
    if iface0 is not None:
        cur = getattr(iface0, "_auto_switch_fsm_before_control", None)
        if cur:
            try:
                iface0._auto_switch_fsm_before_control = False  # noqa: SLF001
                print("[OK] 已关闭 send_target 的自动 FSM 切换（保护 BASE_LOCK）")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 关闭自动 FSM 切换失败: {exc}")

    # ---- 不再复位仿真 ----
    #
    # 分工约定：**仿真的开关/复位由人负责**，推理脚本只负责「你开好之后让机器人动」。
    #
    # 这里原本会调 reset_simulation_state()（/set_simulation_state + 等物体稳定）。
    # 它有两处副作用，都很难看：
    #   1) 仿真时间从 0 重新走，而接口进程已经收过上一轮的 TF（stamp 16.6s）。
    #      ros2_robot_interface 新版有 clock-regression 检测
    #      （CLOCK_REGRESSION_THRESHOLD_SEC=1.0），于是判定"仿真重启"并重建 TF buffer，
    #      重建窗口期内 tf2 对每条旧 TF 打 TF_OLD_DATA —— 就是那段刷屏。
    #   2) 会把人手动摆好的场景状态冲掉。
    #
    # 所以改成只**等**场景可用（等到末端位姿有稳定发布为止），不写任何仿真状态。
    motion_cfg = MOTION_CFG if MOTION_CFG is not None else ROBOT_CFG
    # `--no-base-lock` 时跳过锁底盘（全身控制可能自行挪底盘）。
    # 注意这个变量在运行期守护里也要用（见下面的 _guard_base_lock），
    # 所以必须在 bringup 之前定义好 —— 之前重构时漏了，运行时 NameError。
    want_lock = not getattr(args, "no_base_lock", False)
    if not want_lock:
        print("[info] --no-base-lock: 跳过底盘锁定")
    print("Waiting for simulation to be ready (只读等待，不复位仿真)...")
    _wait_for_scene_ready(robot, timeout_s=15.0)

    lock_ok = bringup_control(robot, args, sim_time,
                              motion_cfg=motion_cfg, want_lock=want_lock)

    print("[OK] FSM=OCS2, arms enabled, base locked — starting inference loop...")
    if not lock_ok:
        print("[WARN] 锁底盘/使能未完全确认，先跑几帧观察；若底盘漂移请中断。")

    # ---- 运行期守护：底盘锁被冲掉就补回来 ----
    #
    # 即使启动时锁成功了，运控仍可能在某些时刻重置全身模式
    # （resetHumanoidModeToFsmDefault，见 WbcCtrlComponent.cpp:299 / Ocs2WbcController.cpp:100）。
    # 每帧检查一次 current_state，发现 base_state 不是 LOCKED 就立刻补发 BASE_LOCK。
    # 检查本身是本地缓存读（wbc_state 由订阅回调更新），不阻塞控制循环。
    iface = robot.ros2_interface      # bringup_control 用的是自己的局部 iface
    guard_checker = getattr(iface, "mode_commands_match_wbc_state", None)
    _guard = {"count": 0, "last_relock": 0.0}

    def _guard_base_lock() -> None:
        if guard_checker is None:
            return
        _guard["count"] += 1
        if _guard["count"] % 10:          # 每 10 帧查一次（20Hz 下即 0.5s）
            return
        # 一起守：底盘锁 + HOME 约束关闭。两者任何一个被运控重置，
        # 都会让手臂在无指令的情况下被拉走。
        want = (["BASE_LOCK"] if want_lock else []) + ["HOME_JOINT_OFF"]
        try:
            if guard_checker(want) is True:
                return
        except Exception:                  # noqa: BLE001
            return
        now = sim_time.now_seconds()
        if now - _guard["last_relock"] < 1.0:   # 限频，别刷爆运控队列
            return
        _guard["last_relock"] = now
        for c in want:
            try:
                iface.send_mode_command(c)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 补发 {c} 失败: {exc}")
        print(f"[warn] 全身模式丢失，已补发 {want}")

    period = 1.0 / args.hz
    # 循环节拍模式：
    #   sim  —— 按**仿真时间**等（默认，与采集时的时间轴语义一致）
    #   wall —— 按**墙钟**等（仿真跑不到实时时用它，保证实际吞吐）
    #
    # 关键背景：训练数据是严格 20Hz **仿真时间**（每帧 50ms，见 session.json
    # 的 every×physics_dt）。理想情况下推理也按仿真时间 50ms 走。
    # 但实测闭环只有 1.4Hz（700ms/帧），因为每帧的 get_observation（三路相机
    # async_read 各 300ms 超时）+ 前向 + `sim_time.sleep` 叠加，在 RTF≈0.09
    # 的仿真里被放大了 11 倍 —— 于是动作被拉慢十几倍，闭环必然失败。
    # 开环回放（一次前向、固定 sleep）能成功，正是因为它没这个问题。
    clock_mode = str(getattr(args, "clock", "sim") or "sim").lower()
    period_fn = (lambda: period) if clock_mode == "sim" else (lambda: period)
    now_fn = sim_time.now_seconds if clock_mode == "sim" else time.monotonic

    # 分段计时（--debug 时打印，定位 700ms 到底花在哪）
    _prof = {"obs": 0.0, "infer": 0.0, "send": 0.0, "sleep": 0.0, "n": 0}

    def _report_prof() -> None:
        n = _prof["n"]
        if n < 5:
            return
        tot = sum(_prof[k] for k in ("obs", "infer", "send", "sleep"))
        print(f"[prof] 每帧平均（{n} 帧，{clock_mode} 节拍）: "
              f"obs={_prof['obs']/n*1000:6.1f}ms  "
              f"infer={_prof['infer']/n*1000:6.1f}ms  "
              f"send={_prof['send']/n*1000:5.1f}ms  "
              f"sleep={_prof['sleep']/n*1000:6.1f}ms  "
              f"合计={tot/n*1000:6.1f}ms → {n/tot:.2f} Hz"
              if tot > 0 else "")
        _prof.update({"obs": 0.0, "infer": 0.0, "send": 0.0, "sleep": 0.0, "n": 0})

    try:
        missing_warned = False
        while True:
            t0 = now_fn()
            _guard_base_lock()

            _t_obs = time.monotonic()
            raw_obs = robot.get_observation()
            _prof["obs"] += time.monotonic() - _t_obs

            _maybe_inject_rot6d_in_obs(raw_obs, meta)
            if not missing_warned and USE_STATE_INPUT and "observation.state" in meta.features:
                expected = meta.features.get("observation.state", {}).get("names", [])
                missing = [name for name in expected if name not in raw_obs]
                if missing:
                    print(f"[WARN] Missing observation.state keys: {missing}")
                missing_warned = True

            _t_inf = time.monotonic()
            action_vec = infer_action_vec(
                policy=policy,
                meta=meta,
                policy_image_feats=policy_image_feats,
                raw_obs=raw_obs,
                device=args.device,
                hz=args.hz,
                sleep_fn=sim_time.sleep,
                preprocessor=_pre,
                postprocessor=_post,
            )
            _prof["infer"] += time.monotonic() - _t_inf
            if action_vec is None:
                continue

            cmd = build_cmd_from_action_vec(
                meta, action_vec, raw_obs, args, gripper_idx, right_gripper_idx
            )
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
            _t_send = time.monotonic()
            robot.send_action(cmd)
            _prof["send"] += time.monotonic() - _t_send
            _prof["n"] += 1

            elapsed = now_fn() - t0
            sleep_t = period_fn() - elapsed
            if sleep_t > 0:
                _t_sl = time.monotonic()
                if clock_mode == "sim":
                    sim_time.sleep(sleep_t)
                else:
                    time.sleep(sleep_t)
                _prof["sleep"] += time.monotonic() - _t_sl
            if _prof["n"] % 40 == 0:
                _report_prof()
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
