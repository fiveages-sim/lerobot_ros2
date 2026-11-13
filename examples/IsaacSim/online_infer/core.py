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
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot_robot_ros2 import (
    ROS2Robot,
)

from isaac_ros2_sim_common import SimTimeHelper, reset_simulation_state  # pyright: ignore[reportMissingImports]
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


def _refresh_tf_buffer_after_sim_reset(robot: ROS2Robot) -> None:
    """Drop stale TF cache after Isaac STOP/PLAY rewinds sim time."""
    try:
        import tf2_ros
    except Exception as exc:
        print(f"[WARN] tf2_ros unavailable; cannot refresh TF buffer ({exc})")
        return

    iface = getattr(robot, "ros2_interface", None)
    node = getattr(iface, "robot_node", None) if iface is not None else None
    if iface is None or node is None:
        print("[WARN] Robot ROS interface not ready; skipped TF buffer refresh")
        return

    iface.tf_buffer = tf2_ros.Buffer()
    iface.tf_listener = tf2_ros.TransformListener(iface.tf_buffer, node)
    print("[OK] TF buffer refreshed after simulation reset")
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
    camera_config = apply_fps_to_cameras(robot_config.cameras, int(FPS))
    # robot.yaml stays target_command for motion-generation; ACT needs continuous %.
    robot_config = replace(
        robot_config,
        cameras=camera_config,
        gripper_control_mode="position",
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


def _quat_mul_xyzw(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Hamilton product for (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_slerp_xyzw(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation; ``t`` in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        out = out / max(float(np.linalg.norm(out)), 1e-12)
        return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    dot = float(np.clip(dot, -1.0, 1.0))
    theta = float(np.arccos(dot))
    s = float(np.sin(theta))
    w0 = float(np.sin((1.0 - t) * theta) / s)
    w1 = float(np.sin(t * theta) / s)
    out = w0 * a + w1 * b
    return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))


def _gripper_percent_from_joint(pos: float, *, closed: float = -0.02, opened: float = 0.0245) -> float:
    span = opened - closed
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((pos - closed) / span, 0.0, 1.0))


def _inject_collector_pose_aliases(raw_obs: Dict[str, Any], robot: ROS2Robot) -> None:
    """Map ROS2Robot EE keys → G1D collector pose-mode names (left_pos_x, …).

    Training data from data-collector-platform uses ``left_pos_*`` / ``right_pos_*``
    / ``*_gripper``, while ``ROS2Robot.get_observation()`` emits ``left_ee.pos.*``.
    Without this bridge the policy state is all zeros and control is nonsense.
    """
    side_map = (
        ("left", "left_ee", "left_gripper_joint"),
        ("right", "right_ee", "right_gripper_joint"),
    )
    for prefix, ee, grip_joint in side_map:
        for axis in ("x", "y", "z"):
            raw_obs[f"{prefix}_pos_{axis}"] = float(raw_obs.get(f"{ee}.pos.{axis}", 0.0))
        for comp in ("x", "y", "z", "w"):
            default = 1.0 if comp == "w" else 0.0
            raw_obs[f"{prefix}_quat_{comp}"] = float(raw_obs.get(f"{ee}.quat.{comp}", default))

        grip_key = f"{prefix}_gripper.pos"
        if grip_key in raw_obs:
            raw_obs[f"{prefix}_gripper"] = float(raw_obs[grip_key])
        else:
            # Best-effort from /joint_states (Dex1 finger joint).
            js = None
            try:
                js = robot.ros2_interface.get_joint_state()
            except Exception:
                js = None
            val = 1.0
            if isinstance(js, dict) and grip_joint in (js.get("names") or []):
                idx = list(js["names"]).index(grip_joint)
                val = _gripper_percent_from_joint(float(js["positions"][idx]))
            raw_obs[f"{prefix}_gripper"] = val

    # Collector pads unused dims with zeros; keep schema length without warnings.
    for i in range(16):
        raw_obs.setdefault(f"pad_{i}", 0.0)


# Dataset action |dpos| max ≈ 6 mm/step; clamp OOD ACT spikes that yank into self-collision.
_MAX_DPOS_M = 0.008
# Open-loop EE reference may lead measured pose (teleop targets led by ~2–4 cm).
# Cap the lead so a stuck arm cannot accumulate an unreachable waypoint.
_MAX_EE_LEAD_M = 0.05
# Training dquat is near-identity (w≈1, xyz std < 2e-3). Hold measured orientation unless
# the predicted delta is clearly intentional — noisy dquat was flinging wrists into the torso.
_MIN_DQUAT_ANGLE = 0.02  # rad (~1.1°)
# Match pick_tube_full_new.yaml modes for ACT:
#   BASE_LOCK         → chassis locked (YAML)
#   BODY_RELATIVE     → same as YAML default: WBC may raise LZ_mt/LZ_it while
#                       tracking EE (demos: LZ 0→~0.03m by grasp, then higher).
#                       YAML never BODY_LOCKs; locking lift made the arm float
#                       above the tube / stall short in x.
#   LEFT_ARM_DISABLE  → YAML
#   HOME_JOINT_OFF    → ACT-only: avoid HOME droop on OCS2 enter
# Apply in FSM=HOLD before OCS2 so the default mobile-base mode never runs first.
_RIGHT_GRASP_CLOSED = 0.25
_RIGHT_GRASP_CLOSED_STEPS = 8
# Demo approach (ep0): z rises over the rack (→~0.94) while pitch slerps 0°→30°
# as x goes 2.10→1.94, then settles to grasp ≈ (1.916, 0.236, 0.930) and closes.
# Prior ACT clamps forced z≤0.915 (into the rack) and snapped full -30° at x≤2.02.
_GRASP_X = 1.916
_GRASP_Y = 0.236
_GRASP_Z = 0.930
_GRASP_QUAT_XYZW = (-0.260955, 0.000219, 0.965351, 0.000441)
_APPROACH_X_START = 2.10  # begin pitch slerp (demo ~1.6°)
_APPROACH_X_END = 1.94  # full ~30° pitch (demo)
_PRE_GRASP_Z_MIN = 0.880
_PRE_GRASP_Z_MAX = 0.945  # clear the rack like demos (was 0.915 → collisions)
# Close as soon as pitch is full / at the tube face — waiting until 1.925 with an
# open gripper was plowing the tube backward for ~1.5 s.
_PRE_GRASP_X_MAX = 1.940
_PRE_GRASP_X_MIN = 1.900
_PRE_GRASP_Y_MIN = 0.228
_PRE_GRASP_Y_MAX = 0.245
_MAX_STUCK_LEAD_M = 0.02
_EE_HOLD_SETTLE_S = 1.0
_FORCE_CLOSE_GRIPPER = 0.0
_POST_GRASP_Z_LIFT = 0.20  # YAML single_arm.move_relative +0.20 world Z
# ACT post-grasp |dpos_z|≈0.05 mm/step → ~2 min lift. Script ~4 mm/step (~0.12 m/s).
_LIFT_STEP_M = 0.004
_LIFT_DONE_TOL_M = 0.025
_GRASP_SETTLE_STEPS = 6  # hold closed at grasp pose before lifting
_LEFT_ENABLE_WAIT_S = 10.0  # OCS2 arm-policy hold often needs >3 s
# Demo world chest handover (ep0 at left-close): right brings tube in, left meets it.
# Pinning right at pick XY after lift left ACT hovering — must script this phase.
_HANDOVER_RIGHT_XYZ = (2.072, 0.204, 1.096)
_HANDOVER_RIGHT_QUAT = (-0.224069, -0.130218, 0.843456, -0.470550)
_HANDOVER_LEFT_XYZ = (2.059, 0.002, 1.041)
_HANDOVER_LEFT_QUAT = (-0.114797, 0.065468, 0.859709, 0.493392)
_HANDOVER_STEP_M = 0.006
_HANDOVER_ARRIVE_M = 0.03
_HANDOVER_LEFT_CLOSE_STEPS = 10
_ACT_PICK_MODE_COMMANDS = [
    "ARMS_INDEPENDENT",
    "BASE_LOCK",
    "BODY_RELATIVE",  # allow lift UP like motion-gen demos / YAML (not BODY_LOCK)
    "LEFT_ARM_DISABLE",
    "HOME_JOINT_OFF",
]


def _apply_mode_commands(
    iface,
    commands: list[str],
    *,
    wait_timeout: float = 5.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Publish /mode_command strings the same way task YAML collection does."""
    for command in commands:
        iface.send_mode_command(command)
        print(f"[OK] mode_command={command}")
        sleep_fn(0.05)
    wait = getattr(iface, "wait_until_mode_commands_applied", None)
    if wait is None:
        sleep_fn(0.2)
        return
    try:
        ok = wait(commands, timeout=wait_timeout, sleep_fn=sleep_fn)
    except Exception as exc:
        print(f"[WARN] wait_until_mode_commands_applied failed: {exc}")
        return
    if ok is False:
        print(f"[WARN] mode commands not confirmed within {wait_timeout:.1f}s: {commands}")
    else:
        print(f"[OK] mode commands confirmed: {commands}")
    # Surface WBC flags so chassis/lift leaks are obvious in the ACT log.
    try:
        st = getattr(iface, "wbc_state", None)
        if st is not None:
            print(
                "[info] WBC state after modes: "
                f"base={int(st.base_state)} body={int(st.body_state)} "
                f"left_arm={int(st.left_arm_state)} right_arm={int(st.right_arm_state)} "
                f"bimanual={int(st.bimanual_state)}"
            )
    except Exception as exc:
        print(f"[WARN] could not read wbc_state: {exc}")


def _publish_ee_hold_current(robot: ROS2Robot, sides: tuple[str, ...] = ("left", "right")) -> None:
    """Re-publish each arm's measured pose as the continuous target (no motion)."""
    iface = robot.ros2_interface
    for side in sides:
        handler = getattr(iface, f"{side}_arm_handler", None)
        if handler is None or not hasattr(handler, "get_pose") or not hasattr(handler, "send_target"):
            continue
        pose = handler.get_pose()
        if pose is None:
            continue
        try:
            handler.send_target(pose)
        except Exception:
            pass


def _step_pose_toward(
    ee_ref: Dict[str, Dict[str, np.ndarray]],
    side: str,
    target_xyz: tuple[float, float, float],
    target_quat: tuple[float, float, float, float],
    *,
    step_m: float,
    measured: np.ndarray | None = None,
) -> float:
    """Advance open-loop ``ee_ref[side]`` one step toward a world pose. Returns remaining distance."""
    if side not in ee_ref:
        return float("inf")
    cur = ee_ref[side]["pos"]
    tgt = np.asarray(target_xyz, dtype=np.float64)
    delta = tgt - cur
    dist = float(np.linalg.norm(delta))
    if dist > 1e-9:
        cur[:] = cur + delta * (min(step_m, dist) / dist)
    if measured is not None:
        lead = cur - measured
        lead_n = float(np.linalg.norm(lead))
        if lead_n > _MAX_EE_LEAD_M and lead_n > 1e-9:
            cur[:] = measured + lead * (_MAX_EE_LEAD_M / lead_n)
    q0 = tuple(float(v) for v in ee_ref[side]["quat"])
    # Blend orientation with progress of this step (small slerp toward target).
    alpha = 1.0 if dist <= step_m else float(np.clip(step_m / max(dist, 1e-6), 0.05, 0.35))
    q_cmd = _quat_slerp_xyzw(q0, target_quat, alpha)
    ee_ref[side]["quat"] = np.asarray(q_cmd, dtype=np.float64)
    return float(np.linalg.norm(tgt - cur))


def _apply_ee_ref_to_cmd(
    cmd: Dict[str, float],
    ee_ref: Dict[str, Dict[str, np.ndarray]],
    side: str,
    *,
    gripper: float | None = None,
) -> None:
    ee = f"{side}_ee"
    if side not in ee_ref:
        return
    pos = ee_ref[side]["pos"]
    quat = ee_ref[side]["quat"]
    cmd[f"{ee}.pos.x"] = float(pos[0])
    cmd[f"{ee}.pos.y"] = float(pos[1])
    cmd[f"{ee}.pos.z"] = float(pos[2])
    cmd[f"{ee}.quat.x"] = float(quat[0])
    cmd[f"{ee}.quat.y"] = float(quat[1])
    cmd[f"{ee}.quat.z"] = float(quat[2])
    cmd[f"{ee}.quat.w"] = float(quat[3])
    if gripper is not None:
        cmd[f"{side}_gripper.pos"] = float(gripper)


def _clamp_pre_grasp_right_cmd(
    cmd: Dict[str, float],
    ee_ref: Dict[str, Dict[str, np.ndarray]],
    raw_obs: Dict[str, Any],
    action_vec: np.ndarray,
    *,
    grasp_latched: bool = False,
    lift_target_z: float | None = None,
    freeze_xy: tuple[float, float] | None = None,
    handover_active: bool = False,
) -> bool:
    """Shape right EE toward the demo approach corridor until a real grasp.

    Returns True when the right gripper should stay closed (in-window or latched).
    When ``lift_target_z`` is set, ignore ACT z and ramp the open-loop ref up
    (scripted YAML-style +0.20 m lift — ACT post-grasp deltas are ~0.05 mm/step).
    When ``handover_active``, leave right pose alone (caller scripts chest move).
    """
    del action_vec  # policy gripper unused; we gate close ourselves
    if "right_ee.pos.x" not in cmd:
        return False
    if handover_active:
        cmd["right_gripper.pos"] = _FORCE_CLOSE_GRIPPER
        return True

    rx = float(raw_obs.get("right_ee.pos.x", 2.1))
    ry = float(raw_obs.get("right_ee.pos.y", 0.27))
    rz = float(raw_obs.get("right_ee.pos.z", 0.9))

    # Remember approach-start orientation once (for pitch slerp).
    if "right" in ee_ref and "approach_quat" not in ee_ref["right"]:
        ee_ref["right"]["approach_quat"] = np.asarray(ee_ref["right"]["quat"], dtype=np.float64).copy()

    # Pitch slerp with -x progress (demo: 0°@2.10 → 30°@1.94). No hard snap.
    if "right" in ee_ref and "approach_quat" in ee_ref["right"]:
        q0 = tuple(float(v) for v in ee_ref["right"]["approach_quat"])
        alpha = (_APPROACH_X_START - rx) / max(_APPROACH_X_START - _APPROACH_X_END, 1e-6)
        q_cmd = _quat_slerp_xyzw(q0, _GRASP_QUAT_XYZW, alpha)
        cmd["right_ee.quat.x"], cmd["right_ee.quat.y"], cmd["right_ee.quat.z"], cmd["right_ee.quat.w"] = q_cmd
        ee_ref["right"]["quat"] = np.asarray(q_cmd, dtype=np.float64)

    y_cmd = float(np.clip(float(cmd.get("right_ee.pos.y", _GRASP_Y)), _PRE_GRASP_Y_MIN, _PRE_GRASP_Y_MAX))
    if rx <= _APPROACH_X_START:
        y_cmd = float(np.clip(0.6 * y_cmd + 0.4 * _GRASP_Y, _PRE_GRASP_Y_MIN, _PRE_GRASP_Y_MAX))

    x_cmd = float(np.clip(float(cmd["right_ee.pos.x"]), _PRE_GRASP_X_MIN, 2.15))
    z_raw = float(cmd.get("right_ee.pos.z", rz))
    # Freeze XY at first contact — commanding _GRASP_X=1.916 from x=1.94 shoves the tube.
    hold_x, hold_y = (freeze_xy if freeze_xy is not None else (rx, ry))

    if lift_target_z is not None and "right" in ee_ref:
        x_cmd = float(hold_x)
        y_cmd = float(hold_y)
        z_cmd = float(min(float(ee_ref["right"]["pos"][2]) + _LIFT_STEP_M, float(lift_target_z)))
        z_cmd = float(max(z_cmd, rz + 0.005))
        z_cmd = float(min(z_cmd, float(lift_target_z)))
    elif grasp_latched:
        x_cmd = float(hold_x)
        y_cmd = float(hold_y)
        z_cmd = float(_GRASP_Z)
    elif rx <= _PRE_GRASP_X_MAX:
        # Close in place — do not drive further -x into the tube.
        x_cmd = float(rx)
        y_cmd = float(np.clip(ry, _PRE_GRASP_Y_MIN, _PRE_GRASP_Y_MAX))
        z_cmd = _GRASP_Z
        if rz > _GRASP_Z + 0.01:
            z_cmd = _GRASP_Z - 0.005
    else:
        z_cmd = float(np.clip(max(z_raw, 0.90), _PRE_GRASP_Z_MIN, _PRE_GRASP_Z_MAX))

    if "right" in ee_ref:
        lead_x = float(ee_ref["right"]["pos"][0]) - rx
        if (
            abs(lead_x) > _MAX_STUCK_LEAD_M
            and rx > _PRE_GRASP_X_MAX
            and not grasp_latched
            and lift_target_z is None
        ):
            ee_ref["right"]["pos"][0] = rx + float(np.sign(lead_x)) * _MAX_STUCK_LEAD_M
            x_cmd = float(ee_ref["right"]["pos"][0])
        if rx <= _PRE_GRASP_X_MAX or grasp_latched or lift_target_z is not None:
            ee_ref["right"]["pos"][0] = x_cmd
        else:
            ee_ref["right"]["pos"][0] = float(
                np.clip(float(ee_ref["right"]["pos"][0]), _PRE_GRASP_X_MIN, 2.15)
            )
        ee_ref["right"]["pos"][1] = y_cmd
        ee_ref["right"]["pos"][2] = z_cmd

    cmd["right_ee.pos.x"] = x_cmd
    cmd["right_ee.pos.y"] = y_cmd
    cmd["right_ee.pos.z"] = z_cmd

    if grasp_latched or lift_target_z is not None:
        cmd["right_gripper.pos"] = _FORCE_CLOSE_GRIPPER
        return True

    near_tube = (
        rx <= _PRE_GRASP_X_MAX
        and abs(rz - _GRASP_Z) <= 0.025
        and abs(ry - _GRASP_Y) <= 0.015
    )
    if not near_tube:
        cmd["right_gripper.pos"] = 1.0
        return False
    cmd["right_gripper.pos"] = _FORCE_CLOSE_GRIPPER
    return True


def _strip_arm_from_cmd(cmd: Dict[str, float], side: str) -> None:
    """Drop EE pose (+ optional gripper) keys so send_action won't drive that arm."""
    prefixes = (f"{side}_ee.",)
    for key in list(cmd.keys()):
        if key.startswith(prefixes) or key == f"{side}_gripper.pos":
            cmd.pop(key, None)


def _seed_ee_ref_from_obs(ee_ref: Dict[str, Dict[str, np.ndarray]], raw_obs: Dict[str, Any], side: str) -> None:
    """Initialize open-loop EE reference from the current measured pose (once per side)."""
    ee = f"{side}_ee"
    ee_ref[side] = {
        "pos": np.asarray(
            [
                float(raw_obs.get(f"{ee}.pos.x", 0.0)),
                float(raw_obs.get(f"{ee}.pos.y", 0.0)),
                float(raw_obs.get(f"{ee}.pos.z", 0.0)),
            ],
            dtype=np.float64,
        ),
        "quat": np.asarray(
            [
                float(raw_obs.get(f"{ee}.quat.x", 0.0)),
                float(raw_obs.get(f"{ee}.quat.y", 0.0)),
                float(raw_obs.get(f"{ee}.quat.z", 0.0)),
                float(raw_obs.get(f"{ee}.quat.w", 1.0)),
            ],
            dtype=np.float64,
        ),
    }


def _collector_delta_action_to_robot_cmd(
    action_named: Dict[str, float],
    raw_obs: Dict[str, Any],
    ee_ref: Dict[str, Dict[str, np.ndarray]] | None = None,
) -> Dict[str, float]:
    """Convert collector ``*_dpos_*`` / ``*_dquat_*`` deltas into ROS2Robot EE cmds.

    When ``ee_ref`` is provided, deltas integrate open-loop onto that reference
    (teleop-style leading targets). Re-basing on measured pose every step lets
    competing OCS2 costs (HOME, body RELATIVE) ratchet the hand down when |dpos|
    is only ~1 mm.
    """
    cmd: Dict[str, float] = {}
    for side, ee in (("left", "left_ee"), ("right", "right_ee")):
        if f"{side}_dpos_x" not in action_named and f"{side}_pos_x" not in action_named:
            continue
        # Absolute pose features (rare) or delta features (G1D tube dataset).
        if f"{side}_dpos_x" in action_named:
            dp = np.asarray(
                [
                    float(action_named.get(f"{side}_dpos_x", 0.0)),
                    float(action_named.get(f"{side}_dpos_y", 0.0)),
                    float(action_named.get(f"{side}_dpos_z", 0.0)),
                ],
                dtype=np.float64,
            )
            dp = np.clip(dp, -_MAX_DPOS_M, _MAX_DPOS_M)

            measured = np.asarray(
                [
                    float(raw_obs.get(f"{ee}.pos.x", 0.0)),
                    float(raw_obs.get(f"{ee}.pos.y", 0.0)),
                    float(raw_obs.get(f"{ee}.pos.z", 0.0)),
                ],
                dtype=np.float64,
            )
            if ee_ref is not None:
                if side not in ee_ref:
                    _seed_ee_ref_from_obs(ee_ref, raw_obs, side)
                ref = ee_ref[side]
                ref["pos"] = ref["pos"] + dp
                lead = ref["pos"] - measured
                lead_norm = float(np.linalg.norm(lead))
                if lead_norm > _MAX_EE_LEAD_M and lead_norm > 1e-9:
                    ref["pos"] = measured + lead * (_MAX_EE_LEAD_M / lead_norm)
                target_pos = ref["pos"]
                q_cur = (
                    float(ref["quat"][0]),
                    float(ref["quat"][1]),
                    float(ref["quat"][2]),
                    float(ref["quat"][3]),
                )
            else:
                target_pos = measured + dp
                q_cur = (
                    float(raw_obs.get(f"{ee}.quat.x", 0.0)),
                    float(raw_obs.get(f"{ee}.quat.y", 0.0)),
                    float(raw_obs.get(f"{ee}.quat.z", 0.0)),
                    float(raw_obs.get(f"{ee}.quat.w", 1.0)),
                )

            cmd[f"{ee}.pos.x"] = float(target_pos[0])
            cmd[f"{ee}.pos.y"] = float(target_pos[1])
            cmd[f"{ee}.pos.z"] = float(target_pos[2])

            q_delta = (
                float(action_named.get(f"{side}_dquat_x", 0.0)),
                float(action_named.get(f"{side}_dquat_y", 0.0)),
                float(action_named.get(f"{side}_dquat_z", 0.0)),
                float(action_named.get(f"{side}_dquat_w", 1.0)),
            )
            # q_delta = q_next * conj(q_cur) in converter ⇒ q_next = q_delta * q_cur.
            # Skip near-identity / noisy deltas (w≈1) to keep wrists stable.
            qw = float(np.clip(abs(q_delta[3]), 0.0, 1.0))
            angle = 2.0 * float(np.arccos(qw))
            if angle >= _MIN_DQUAT_ANGLE:
                q_new = _quat_mul_xyzw(q_delta, q_cur)
            else:
                q_new = q_cur
            if ee_ref is not None and side in ee_ref:
                ee_ref[side]["quat"] = np.asarray(q_new, dtype=np.float64)
            cmd[f"{ee}.quat.x"], cmd[f"{ee}.quat.y"], cmd[f"{ee}.quat.z"], cmd[f"{ee}.quat.w"] = q_new
        else:
            cmd[f"{ee}.pos.x"] = float(action_named.get(f"{side}_pos_x", raw_obs.get(f"{ee}.pos.x", 0.0)))
            cmd[f"{ee}.pos.y"] = float(action_named.get(f"{side}_pos_y", raw_obs.get(f"{ee}.pos.y", 0.0)))
            cmd[f"{ee}.pos.z"] = float(action_named.get(f"{side}_pos_z", raw_obs.get(f"{ee}.pos.z", 0.0)))
            cmd[f"{ee}.quat.x"] = float(action_named.get(f"{side}_quat_x", raw_obs.get(f"{ee}.quat.x", 0.0)))
            cmd[f"{ee}.quat.y"] = float(action_named.get(f"{side}_quat_y", raw_obs.get(f"{ee}.quat.y", 0.0)))
            cmd[f"{ee}.quat.z"] = float(action_named.get(f"{side}_quat_z", raw_obs.get(f"{ee}.quat.z", 0.0)))
            cmd[f"{ee}.quat.w"] = float(action_named.get(f"{side}_quat_w", raw_obs.get(f"{ee}.quat.w", 1.0)))

        if f"{side}_gripper" in action_named:
            cmd[f"{side}_gripper.pos"] = float(np.clip(action_named[f"{side}_gripper"], 0.0, 1.0))
    return cmd


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


def init_inference(args):
    """加载训练配置/数据集元信息/策略模型 + MEAN_STD 预/后处理，用于在线推理。"""
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

    # Newer LeRobot ACT checkpoints keep MEAN_STD outside the weights
    # (policy_preprocessor / policy_postprocessor). Skipping them feeds raw
    # state/images and treats z-score actions as meters → arms thrash.
    pretrained_dir = Path(ckpt_path)
    if (pretrained_dir / "pretrained_model").is_dir():
        pretrained_dir = pretrained_dir / "pretrained_model"
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=train_cfg.policy,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
        },
    )
    print(f"[info] Loaded policy preprocessor/postprocessor from {pretrained_dir}")

    policy.to(args.device)
    policy.eval()
    policy.reset()
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()
    print("[info] Policy reset, action queue initialized")

    action_names = meta.features.get("action", {}).get("names", [])
    gripper_idx = action_names.index("left_gripper.pos") if "left_gripper.pos" in action_names else None

    return train_cfg, meta, policy, preprocessor, postprocessor, policy_image_feats, gripper_idx


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
    """执行一次策略推理并返回物理单位动作向量（经 postprocessor unnormalize）。"""
    batch = {}
    if USE_STATE_INPUT and "observation.state" in meta.features:
        state = torch.from_numpy(build_state_vector(meta, raw_obs)).unsqueeze(0)
        batch["observation.state"] = state

    # Dataset intervention is always 1 during teleop demos; keep policy input in-distribution.
    if "observation.intervention" in getattr(getattr(policy, "config", None), "input_features", {}):
        batch["observation.intervention"] = torch.ones(1, 1, dtype=torch.float32)

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

    if preprocessor is not None:
        batch = preprocessor(batch)

    with torch.inference_mode():
        action = policy.select_action(batch)

    if postprocessor is not None:
        action = postprocessor(action)

    action_vec = action.squeeze(0).detach().cpu().numpy()
    return action_vec


def build_cmd_from_action_vec(
    meta: LeRobotDatasetMetadata,
    action_vec: np.ndarray,
    raw_obs: Dict[str, float],
    args,
    gripper_idx: int | None,
    ee_ref: Dict[str, Dict[str, np.ndarray]] | None = None,
) -> Dict[str, float]:
    """将动作向量转换为机器人命令字典，并应用可选覆盖。"""
    named = build_action_dict(meta, action_vec)

    # G1D collector pose-mode: actions are left_dpos_* / right_dpos_* deltas.
    if any(k.endswith("_dpos_x") or k.endswith("_pos_x") for k in named):
        cmd = _collector_delta_action_to_robot_cmd(named, raw_obs, ee_ref=ee_ref)
    else:
        cmd = dict(named)
        if args.relative:
            cmd["left_ee.pos.x"] = float(cmd.get("left_ee.pos.x", 0.0)) + float(raw_obs.get("left_ee.pos.x", 0.0))
            cmd["left_ee.pos.y"] = float(cmd.get("left_ee.pos.y", 0.0)) + float(raw_obs.get("left_ee.pos.y", 0.0))
            cmd["left_ee.pos.z"] = float(cmd.get("left_ee.pos.z", 0.0)) + float(raw_obs.get("left_ee.pos.z", 0.0))

    if gripper_idx is not None:
        g = float(np.clip(action_vec[gripper_idx], 0.0, 1.0))
        if args.fixed_gripper is not None:
            g = float(np.clip(args.fixed_gripper, 0.0, 1.0))
        elif args.auto_close_z is not None:
            curr_z = raw_obs.get("left_ee.pos.z")
            if curr_z is not None and curr_z <= args.auto_close_z:
                g = 0.0
        # Only override left if dual-arm collector cmd did not already set grippers.
        cmd.setdefault("left_gripper.pos", g)

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
    _train_cfg, meta, policy, preprocessor, postprocessor, policy_image_feats, gripper_idx = init_inference(
        args
    )

    robot = build_robot()
    sim_time = SimTimeHelper()
    robot_connected = False

    def shutdown_handler(sig, frame) -> None:
        nonlocal robot_connected
        print("\nInterrupt received, shutting down...")
        try:
            if robot_connected:
                try:
                    robot.ros2_interface.send_fsm_command(FSM_HOLD)
                except Exception:
                    pass
                try:
                    robot.disconnect()
                except Exception:
                    pass
                robot_connected = False
        finally:
            sim_time.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    if "object_prim_path" not in PICK_PLACE_FLOW_OVERRIDES:
        raise RuntimeError("PICK_PLACE_FLOW_OVERRIDES.object_prim_path is required for reset")
    motion_cfg = MOTION_CFG if MOTION_CFG is not None else ROBOT_CFG
    if motion_cfg is None:
        raise RuntimeError("MOTION_CFG is not configured. Run via examples/IsaacSim/inference.py.")

    # Reset BEFORE connect. Connecting first fills the TF cache with large sim
    # stamps; STOP/PLAY then rewinds /clock → endless TF_OLD_DATA and we never
    # reach the post-reset TF refresh (stuck inside settle waits).
    print("Resetting simulation state...")
    reset_simulation_state(
        str(PICK_PLACE_FLOW_OVERRIDES["object_prim_path"]).strip(),
        post_reset_wait=float(getattr(motion_cfg, "post_reset_wait", 3.0) or 3.0),
        # Wall clock: /clock was just rewound; sim-time sleep is fragile here.
        sleep_fn=time.sleep,
    )
    print("[OK] Post-reset settle done (wall clock)")

    print("Connecting to robot...")
    robot.connect()
    robot_connected = True
    print("[OK] Robot connected")
    # Pose-mode ACT does not use W2-style body_joint* aliases; clear to stop spam.
    try:
        robot.config.ros2_interface.joint_names = []
    except Exception:
        pass
    # Safety if anything still held a stale buffer from a prior session.
    _refresh_tf_buffer_after_sim_reset(robot)
    time.sleep(0.5)

    # Collision→HOLD then ArmHandler auto-switch back to OCS2 caused the jerky
    # HOLD↔OCS2 thrash. Inference owns FSM; do not auto-reenter after a collision.
    try:
        robot.ros2_interface.config.auto_switch_fsm_before_control = False
        # Dex1 AdaptiveGripper joint range (matches controller URDF limits).
        robot.ros2_interface.config.gripper_min_position = -0.02
        robot.ros2_interface.config.gripper_max_position = 0.0245
    except Exception:
        pass

    # 1) HOLD → 2) BASE_LOCK + BODY_RELATIVE + left disable → 3) seed EE → 4) OCS2.
    # Entering OCS2 unlocked first was letting HM_DUAL_RELATIVE_MOBILE_BASE
    # roll the chassis before BASE_LOCK landed.
    fsm_delay = float(getattr(motion_cfg, "fsm_switch_delay", 0.5) or 0.5)
    print("FSM → HOLD (preset locks before OCS2) ...")
    robot.ros2_interface.send_fsm_command(FSM_HOLD)
    time.sleep(fsm_delay)

    print(
        "[info] Applying ACT pick modes in HOLD: "
        + " → ".join(_ACT_PICK_MODE_COMMANDS)
    )
    _apply_mode_commands(
        robot.ros2_interface,
        list(_ACT_PICK_MODE_COMMANDS),
        wait_timeout=5.0,
        sleep_fn=time.sleep,
    )

    # Seed continuous targets at the measured pose so OCS2 does not yank to
    # a stale/zero EE goal (arms drooping for ~2 s then recovering).
    _publish_ee_hold_current(robot, ("left", "right"))
    time.sleep(0.1)

    print("Switching FSM: HOLD -> OCS2 (locks already set) ...")
    robot.ros2_interface.send_fsm_command(FSM_OCS2)
    time.sleep(fsm_delay)
    print("[OK] FSM switched to OCS2")

    # Re-assert after enter in case activate cleared the preset.
    _apply_mode_commands(
        robot.ros2_interface,
        list(_ACT_PICK_MODE_COMMANDS),
        wait_timeout=5.0,
        sleep_fn=time.sleep,
    )

    settle_steps = max(1, int(round(_EE_HOLD_SETTLE_S * float(args.hz))))
    print(
        f"[info] Holding EE at current pose for {settle_steps} steps "
        f"({_EE_HOLD_SETTLE_S:.1f}s) before ACT ..."
    )
    for _ in range(settle_steps):
        _publish_ee_hold_current(robot, ("right",))  # left stays disabled
        time.sleep(1.0 / float(args.hz))
    print("[OK] EE hold settle done")

    print(
        "[info] Bridging collector pose-mode features "
        "(left_pos_*/left_dpos_* ↔ left_ee.pos.*) for this dataset"
    )
    print(
        f"[info] Open-loop EE deltas (lead≤{_MAX_EE_LEAD_M*100:.0f}cm); "
        f"clamp ±{_MAX_DPOS_M*1000:.0f}mm/step; "
        f"pre-grasp corridor x∈[{_PRE_GRASP_X_MIN:.2f},{_PRE_GRASP_X_MAX:.2f}] "
        f"y∈[{_PRE_GRASP_Y_MIN:.2f},{_PRE_GRASP_Y_MAX:.2f}] "
        f"z∈[{_PRE_GRASP_Z_MIN:.3f},{_PRE_GRASP_Z_MAX:.3f}] (clear rack), "
        f"pitch slerp x∈[{_APPROACH_X_END:.2f},{_APPROACH_X_START:.2f}] → demo 30°; "
        f"close in-place at x≤{_PRE_GRASP_X_MAX:.3f}, "
        f"scripted lift +{_POST_GRASP_Z_LIFT:.2f}m, then chest handover "
        f"R→{_HANDOVER_RIGHT_XYZ} L→{_HANDOVER_LEFT_XYZ}"
    )

    # Demos start with grippers open (1.0). Open both before the policy loop.
    try:
        for side in ("left", "right"):
            handler = getattr(robot.ros2_interface, f"{side}_gripper_handler", None)
            if handler is None:
                continue
            if hasattr(handler, "send_position_percent"):
                handler.send_position_percent(1.0)
            elif hasattr(handler, "send_target_command"):
                handler.send_target_command(1)
        time.sleep(0.5)
        print("[OK] Grippers commanded open before ACT loop")
    except Exception as exc:
        print(f"[WARN] Failed to pre-open grippers: {exc}")

    period = 1.0 / args.hz
    # Open-loop EE setpoint per side (seeded on first delta for that side).
    ee_ref: Dict[str, Dict[str, np.ndarray]] = {}
    try:
        missing_warned = False
        step_i = 0
        hold_skips = 0
        left_arm_enabled = False
        right_grasp_latched = False
        right_lift_target_z: float | None = None
        right_lift_done = False
        grasp_settle_steps = 0
        right_closed_streak = 0
        grasp_freeze_xy: tuple[float, float] | None = None
        # handover: None → "right_chest" → "left_approach" → "left_close" → "done"
        handover_phase: str | None = None
        left_close_streak = 0
        while True:
            t0 = sim_time.now_seconds()
            raw_obs = robot.get_observation()
            _inject_collector_pose_aliases(raw_obs, robot)
            _maybe_inject_rot6d_in_obs(raw_obs, meta)
            if not missing_warned and USE_STATE_INPUT and "observation.state" in meta.features:
                expected = meta.features.get("observation.state", {}).get("names", [])
                missing = [name for name in expected if name not in raw_obs]
                if missing:
                    print(f"[WARN] Missing observation.state keys: {missing}")
                else:
                    print("[OK] observation.state keys matched dataset schema")
                lx, ly, lz = (
                    float(raw_obs.get("left_pos_x", 0.0)),
                    float(raw_obs.get("left_pos_y", 0.0)),
                    float(raw_obs.get("left_pos_z", 0.0)),
                )
                # Demo start ≈ (2.11, -0.03, 0.88). Far away ⇒ policy is OOD.
                demo_err = float(np.linalg.norm([lx - 2.107, ly + 0.031, lz - 0.885]))
                print(
                    f"[info] Start left EE world=({lx:.3f},{ly:.3f},{lz:.3f}) "
                    f"vs demo≈(2.107,-0.031,0.885) err={demo_err:.3f}m"
                )
                if demo_err > 0.35:
                    print(
                        "[WARN] EE start pose far from demo home — ACT will jerk. "
                        "Stop inference, Stop/Play Isaac (or re-run so reset restores home), then retry."
                    )
                missing_warned = True

            # If OCS2 tripped into HOLD on self-collision, do not keep pushing targets.
            try:
                fsm = int(robot.ros2_interface.get_fsm_state())
            except Exception:
                fsm = int(FSM_OCS2)
            if fsm == int(FSM_HOLD):
                hold_skips += 1
                if hold_skips == 1 or hold_skips % 30 == 0:
                    print(
                        f"[WARN] FSM is HOLD (likely self-collision) — pausing ACT commands "
                        f"(skips={hold_skips}). Reset sim / re-home before continuing."
                    )
                sleep_fn = sim_time.sleep
                sleep_fn(period)
                continue
            hold_skips = 0

            action_vec = infer_action_vec(
                policy=policy,
                meta=meta,
                policy_image_feats=policy_image_feats,
                raw_obs=raw_obs,
                device=args.device,
                hz=args.hz,
                sleep_fn=sim_time.sleep,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            if action_vec is None:
                continue

            cmd = build_cmd_from_action_vec(
                meta, action_vec, raw_obs, args, gripper_idx, ee_ref=ee_ref
            )
            _maybe_convert_rot6d_action(cmd)

            rx = float(raw_obs.get("right_ee.pos.x", 2.1))
            ry = float(raw_obs.get("right_ee.pos.y", 0.27))
            rz = float(raw_obs.get("right_ee.pos.z", 0.9))
            lx = float(raw_obs.get("left_ee.pos.x", 2.1))
            ly = float(raw_obs.get("left_ee.pos.y", -0.05))
            lz = float(raw_obs.get("left_ee.pos.z", 0.9))
            handover_active = handover_phase is not None
            right_grasped = _clamp_pre_grasp_right_cmd(
                cmd,
                ee_ref,
                raw_obs,
                action_vec,
                grasp_latched=right_grasp_latched,
                lift_target_z=None if handover_active else right_lift_target_z,
                freeze_xy=grasp_freeze_xy,
                handover_active=handover_active,
            )
            if right_grasped and not right_grasp_latched:
                right_grasp_latched = True
                grasp_settle_steps = 0
                grasp_freeze_xy = (float(rx), float(ry))
                if "right" in ee_ref:
                    ee_ref["right"]["pos"][0] = float(rx)
                    ee_ref["right"]["pos"][1] = float(ry)
                print(
                    f"[OK] Right grasp close at xyz=({rx:.3f},{ry:.3f},{rz:.3f}) — "
                    f"freeze XY, settle {_GRASP_SETTLE_STEPS} steps, then +{_POST_GRASP_Z_LIFT:.2f}m lift"
                )

            # After close: settle → lift → enable left → scripted chest handover.
            if (
                right_grasp_latched
                and right_lift_target_z is None
                and not right_lift_done
                and not handover_active
            ):
                grasp_settle_steps += 1
                if grasp_settle_steps >= _GRASP_SETTLE_STEPS:
                    right_lift_target_z = float(rz) + _POST_GRASP_Z_LIFT
                    if "right" in ee_ref:
                        ee_ref["right"]["pos"][2] = float(rz)
                    print(
                        f"[info] Scripted right EE lift → z={right_lift_target_z:.3f} "
                        f"({_LIFT_STEP_M*1000:.0f} mm/step)"
                    )
            if right_lift_target_z is not None and not right_lift_done and not handover_active:
                if rz >= right_lift_target_z - _LIFT_DONE_TOL_M:
                    right_lift_done = True
                    print(f"[OK] Right lift done (z={rz:.3f}) — enabling left for handover")

            right_g = float(cmd.get("right_gripper.pos", 0.0 if right_grasp_latched else 1.0))
            if not left_arm_enabled:
                if right_lift_done and right_g <= _RIGHT_GRASP_CLOSED:
                    right_closed_streak += 1
                else:
                    right_closed_streak = 0
                if right_closed_streak >= _RIGHT_GRASP_CLOSED_STEPS:
                    _apply_mode_commands(
                        robot.ros2_interface,
                        ["LEFT_ARM_ENABLE"],
                        wait_timeout=_LEFT_ENABLE_WAIT_S,
                        sleep_fn=sim_time.sleep,
                    )
                    left_ok = False
                    try:
                        st = getattr(robot.ros2_interface, "wbc_state", None)
                        left_ok = st is not None and int(st.left_arm_state) == 1
                    except Exception:
                        left_ok = False
                    if left_ok:
                        left_arm_enabled = True
                        ee_ref.pop("left", None)
                        _seed_ee_ref_from_obs(ee_ref, raw_obs, "left")
                        handover_phase = "right_chest"
                        right_lift_target_z = None  # release lift pin; chest script takes over
                        print(
                            "[OK] LEFT_ARM_ENABLE confirmed — scripted chest handover "
                            f"R→{_HANDOVER_RIGHT_XYZ}"
                        )
                    else:
                        right_closed_streak = 0
                        print(
                            "[WARN] LEFT_ARM_ENABLE not confirmed — will retry; "
                            "keeping left stripped"
                        )
                        _strip_arm_from_cmd(cmd, "left")
                        ee_ref.pop("left", None)
                else:
                    _strip_arm_from_cmd(cmd, "left")
                    ee_ref.pop("left", None)

            # Scripted handover (demos: right to chest, then left approaches tube).
            if handover_phase == "right_chest":
                rem = _step_pose_toward(
                    ee_ref,
                    "right",
                    _HANDOVER_RIGHT_XYZ,
                    _HANDOVER_RIGHT_QUAT,
                    step_m=_HANDOVER_STEP_M,
                    measured=np.asarray([rx, ry, rz], dtype=np.float64),
                )
                _apply_ee_ref_to_cmd(cmd, ee_ref, "right", gripper=_FORCE_CLOSE_GRIPPER)
                # Hold left at current while right travels.
                if "left" in ee_ref:
                    _apply_ee_ref_to_cmd(cmd, ee_ref, "left", gripper=1.0)
                if rem <= _HANDOVER_ARRIVE_M:
                    handover_phase = "left_approach"
                    print(
                        f"[OK] Right at chest (rem={rem:.3f}m) — left approaches "
                        f"{_HANDOVER_LEFT_XYZ}"
                    )
            elif handover_phase == "left_approach":
                # Keep right at chest target.
                rem_r = _step_pose_toward(
                    ee_ref,
                    "right",
                    _HANDOVER_RIGHT_XYZ,
                    _HANDOVER_RIGHT_QUAT,
                    step_m=_HANDOVER_STEP_M,
                    measured=np.asarray([rx, ry, rz], dtype=np.float64),
                )
                _apply_ee_ref_to_cmd(cmd, ee_ref, "right", gripper=_FORCE_CLOSE_GRIPPER)
                if "left" not in ee_ref:
                    _seed_ee_ref_from_obs(ee_ref, raw_obs, "left")
                rem_l = _step_pose_toward(
                    ee_ref,
                    "left",
                    _HANDOVER_LEFT_XYZ,
                    _HANDOVER_LEFT_QUAT,
                    step_m=_HANDOVER_STEP_M,
                    measured=np.asarray([lx, ly, lz], dtype=np.float64),
                )
                _apply_ee_ref_to_cmd(cmd, ee_ref, "left", gripper=1.0)
                if rem_l <= _HANDOVER_ARRIVE_M:
                    handover_phase = "left_close"
                    left_close_streak = 0
                    print(f"[OK] Left at tube (rem={rem_l:.3f}m) — closing left gripper")
            elif handover_phase == "left_close":
                _step_pose_toward(
                    ee_ref,
                    "right",
                    _HANDOVER_RIGHT_XYZ,
                    _HANDOVER_RIGHT_QUAT,
                    step_m=_HANDOVER_STEP_M,
                    measured=np.asarray([rx, ry, rz], dtype=np.float64),
                )
                _step_pose_toward(
                    ee_ref,
                    "left",
                    _HANDOVER_LEFT_XYZ,
                    _HANDOVER_LEFT_QUAT,
                    step_m=_HANDOVER_STEP_M,
                    measured=np.asarray([lx, ly, lz], dtype=np.float64),
                )
                _apply_ee_ref_to_cmd(cmd, ee_ref, "right", gripper=_FORCE_CLOSE_GRIPPER)
                _apply_ee_ref_to_cmd(cmd, ee_ref, "left", gripper=_FORCE_CLOSE_GRIPPER)
                left_close_streak += 1
                if left_close_streak >= _HANDOVER_LEFT_CLOSE_STEPS:
                    # YAML: open right after left has closed.
                    _apply_ee_ref_to_cmd(cmd, ee_ref, "right", gripper=1.0)
                    handover_phase = "done"
                    print("[OK] Handover complete — left closed, right opened")
            elif handover_phase == "done":
                _apply_ee_ref_to_cmd(cmd, ee_ref, "right", gripper=1.0)
                _apply_ee_ref_to_cmd(cmd, ee_ref, "left", gripper=_FORCE_CLOSE_GRIPPER)
            elif not left_arm_enabled:
                pass  # left already stripped above
            else:
                _strip_arm_from_cmd(cmd, "left")

            # Active EE for logging: left during handover approach, else right.
            log_side = (
                "left_ee"
                if handover_phase in ("left_approach", "left_close", "done")
                else "right_ee"
            )
            curr_x = raw_obs.get(f"{log_side}.pos.x")
            curr_y = raw_obs.get(f"{log_side}.pos.y")
            curr_z = raw_obs.get(f"{log_side}.pos.z")

            target_x = cmd.get(f"{log_side}.pos.x", 0.0)
            target_y = cmd.get(f"{log_side}.pos.y", 0.0)
            target_z = cmd.get(f"{log_side}.pos.z", 0.0)

            curr_xyz = None
            target_xyz = None
            delta_xyz = None
            distance = None
            if curr_x is not None and f"{log_side}.pos.x" in cmd:
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
            if step_i < 5 or step_i % 30 == 0:
                dpos = action_vec[8:11] if action_vec is not None and len(action_vec) >= 11 else None
                grip = (
                    float(action_vec[7]),
                    float(cmd.get("right_gripper.pos", float("nan"))),
                )
                print(
                    f"[step {step_i}] phase={handover_phase or ('lift' if right_lift_target_z else 'pick')} "
                    f"grippers(L,R)=("
                    f"{float(cmd.get('left_gripper.pos', grip[0])):.3f},"
                    f"{float(cmd.get('right_gripper.pos', float('nan'))):.3f}) "
                    f"left_arm={'on' if left_arm_enabled else 'off'} "
                    f"R=({rx:.3f},{ry:.3f},{rz:.3f})→("
                    f"{float(cmd.get('right_ee.pos.x', float('nan'))):.3f},"
                    f"{float(cmd.get('right_ee.pos.y', float('nan'))):.3f},"
                    f"{float(cmd.get('right_ee.pos.z', float('nan'))):.3f}) "
                    f"L=({lx:.3f},{ly:.3f},{lz:.3f})→("
                    f"{float(cmd.get('left_ee.pos.x', float('nan'))):.3f},"
                    f"{float(cmd.get('left_ee.pos.y', float('nan'))):.3f},"
                    f"{float(cmd.get('left_ee.pos.z', float('nan'))):.3f})"
                )
                if step_i < 5 and dpos is not None and float(np.linalg.norm(dpos)) > 0.05:
                    print(
                        "[WARN] |right_dpos| still >5cm/step after unnormalize — "
                        "check preprocessor/postprocessor load; demo scale is ~0.3mm/step."
                    )
                # Re-assert BASE_LOCK only. Never re-apply LEFT_ARM_DISABLE after
                # LEFT_ARM_ENABLE (that bug re-disabled the left arm mid-handover).
                if step_i > 0 and step_i % 90 == 0:
                    match = getattr(
                        robot.ros2_interface, "mode_commands_match_wbc_state", None
                    )
                    if callable(match):
                        want = ["BASE_LOCK"]
                        if not left_arm_enabled:
                            want.append("LEFT_ARM_DISABLE")
                        ok = match(want)
                        if ok is False:
                            print(f"[WARN] WBC lost {want} — re-applying")
                            reapply = ["BASE_LOCK", "BODY_RELATIVE"]
                            if not left_arm_enabled:
                                reapply.append("LEFT_ARM_DISABLE")
                            else:
                                reapply.append("LEFT_ARM_ENABLE")
                            _apply_mode_commands(
                                robot.ros2_interface,
                                reapply,
                                wait_timeout=2.0,
                                sleep_fn=sim_time.sleep,
                            )

            step_i += 1
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
