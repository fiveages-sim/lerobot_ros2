#!/usr/bin/env python3
"""
Offline conversion: raw grasp recordings -> LeRobot dataset.

Usage:
  python examples/Cr5_ACT/convert_raw_to_lerobot.py \
  --raw-root /home/fa/lerobot_ros2/raw_dataset/grasp_raw_XXXXXXXX

"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    import torch
    from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "The 'torch' and 'pytorch3d' python packages are required for rotation-6D conversion."
    ) from exc

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw grasp data to LeRobot dataset")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw dataset root (contains meta.json)")
    parser.add_argument("--out-root", type=Path, default=None, help="Output dataset root (optional)")
    parser.add_argument("--copy-points", action="store_true", help="Copy points/ directory if present")
    return parser.parse_args()


def load_meta(raw_root: Path) -> dict:
    meta_path = raw_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    return json.loads(meta_path.read_text())


ROT6D_PREFIX = "end_effector.rot6d"
QUAT_NAMES = [
    "end_effector.orientation.x",
    "end_effector.orientation.y",
    "end_effector.orientation.z",
    "end_effector.orientation.w",
]


def _replace_quat_names(names: list[str]) -> tuple[list[str], bool]:
    if not all(name in names for name in QUAT_NAMES):
        return names, False
    start_idx = min(names.index(n) for n in QUAT_NAMES)
    new_names = [n for n in names if n not in QUAT_NAMES]
    rot6d_names = [f"{ROT6D_PREFIX}_{i}" for i in range(6)]
    for offset, name in enumerate(rot6d_names):
        new_names.insert(start_idx + offset, name)
    return new_names, True


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 elements, got shape {quat.shape}")
    quat = np.atleast_2d(quat)
    # pytorch3d expects (w, x, y, z)
    quat_wxyz = np.stack([quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]], axis=-1)
    rot_mats = quaternion_to_matrix(torch.as_tensor(quat_wxyz, dtype=torch.float32))
    rot6d = matrix_to_rotation_6d(rot_mats).detach().cpu().numpy().astype(np.float32)
    return np.atleast_2d(rot6d)


def build_features(meta: dict) -> tuple[dict, list[str], list[str], bool, bool]:
    state_names = meta.get("state_names") or []
    action_keys = meta.get("action_keys") or []
    include_gripper = bool(meta.get("include_gripper", False))

    state_names, state_rot6d = _replace_quat_names(list(state_names))
    action_names, action_rot6d = _replace_quat_names(list(action_keys))
    if include_gripper and "gripper.position" not in action_names:
        action_names.append("gripper.position")

    state_dim = len(state_names)
    action_dim = len(action_names)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }

    camera_shapes = meta.get("camera_shapes", {})
    for cam_name, shape in camera_shapes.items():
        # shape: [H, W, C]
        if not shape or len(shape) != 3:
            continue
        h, w, c = shape
        features[f"observation.images.{cam_name}_rgb"] = {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": int(h),
                "video.width": int(w),
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": meta.get("fps", 30),
                "video.channels": int(c),
                "has_audio": False,
            },
        }
    return features, state_names, action_names, state_rot6d, action_rot6d


def action_from_next(
    next_ee_pose: np.ndarray,
    next_cmd_gripper: float,
    action_keys: list[str],
    include_gripper: bool,
    use_rot6d: bool,
) -> np.ndarray:
    action_dict = {
        "end_effector.position.x": float(next_ee_pose[0]),
        "end_effector.position.y": float(next_ee_pose[1]),
        "end_effector.position.z": float(next_ee_pose[2]),
    }
    if use_rot6d:
        rot6d = _quat_xyzw_to_rot6d(next_ee_pose[3:7])
        for i in range(6):
            action_dict[f"{ROT6D_PREFIX}_{i}"] = float(rot6d[0, i])
    else:
        action_dict.update(
            {
                "end_effector.orientation.x": float(next_ee_pose[3]),
                "end_effector.orientation.y": float(next_ee_pose[4]),
                "end_effector.orientation.z": float(next_ee_pose[5]),
                "end_effector.orientation.w": float(next_ee_pose[6]),
            }
        )
    if include_gripper:
        action_dict["gripper.position"] = float(np.clip(next_cmd_gripper, 0.0, 1.0))
    values = [action_dict.get(k, 0.0) for k in action_keys]
    if include_gripper:
        values.append(action_dict.get("gripper.position", 0.0))
    return np.asarray(values, dtype=np.float32)


def load_frame_npz(episode_dir: Path, rel_path: str) -> dict:
    data = np.load(episode_dir / rel_path)
    return {
        "observation_state": data["observation_state"],
        "ee_pose": data["ee_pose"],
        "gripper": float(data["gripper"][0]),
        "cmd_gripper": float(data["cmd_gripper"][0]),
        "timestamp": float(data["timestamp"][0]),
    }


def _convert_state_vec(
    state_vec: np.ndarray,
    old_names: list[str],
    new_names: list[str],
    use_rot6d: bool,
) -> np.ndarray:
    if not use_rot6d:
        return state_vec.astype(np.float32, copy=False)
    name_to_val = {name: float(state_vec[i]) for i, name in enumerate(old_names)}
    quat_vals = [name_to_val.get(n) for n in QUAT_NAMES]
    if any(v is None for v in quat_vals):
        return state_vec.astype(np.float32, copy=False)
    rot6d = _quat_xyzw_to_rot6d(np.asarray(quat_vals, dtype=np.float32))
    out = []
    for name in new_names:
        if name.startswith(ROT6D_PREFIX):
            idx = int(name.split("_")[-1])
            out.append(float(rot6d[0, idx]))
        else:
            out.append(float(name_to_val.get(name, 0.0)))
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    meta = load_meta(raw_root)

    features, state_names, action_names, state_rot6d, action_rot6d = build_features(meta)
    out_root = args.out_root.expanduser().resolve() if args.out_root else raw_root.parent / f"{raw_root.name}_lerobot"
    out_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=str(out_root),
        fps=int(meta.get("fps", 30)),
        features=features,
        robot_type=str(meta.get("robot_type", "offline_robot")),
        use_videos=True,
    )

    action_keys = [name for name in action_names if name != "gripper.position"]
    include_gripper = bool(meta.get("include_gripper", False))
    task_name = meta.get("task_name", "grasp")
    original_state_names = meta.get("state_names") or []

    episodes_dir = raw_root / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")])

    for ep_idx, ep_dir in enumerate(episode_dirs):
        frames_path = ep_dir / "frames.jsonl"
        if not frames_path.is_file():
            print(f"[warn] Missing frames.jsonl in {ep_dir}, skipping")
            continue
        frame_meta = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
        if not frame_meta:
            print(f"[warn] Empty episode {ep_dir}, skipping")
            continue

        frames_data = [load_frame_npz(ep_dir, fm["state_path"]) for fm in frame_meta]
        for idx, fm in enumerate(frame_meta):
            state_vec = _convert_state_vec(
                frames_data[idx]["observation_state"],
                original_state_names,
                state_names,
                state_rot6d,
            )
            frame = {"task": task_name, "observation.state": state_vec}

            images = fm.get("images", {})
            for cam_name, rel_path in images.items():
                img = np.load(ep_dir / rel_path)
                frame[f"observation.images.{cam_name}_rgb"] = img

            next_idx = idx + 1 if idx + 1 < len(frames_data) else idx
            next_data = frames_data[next_idx]
            frame["action"] = action_from_next(
                next_ee_pose=next_data["ee_pose"],
                next_cmd_gripper=next_data["cmd_gripper"],
                action_keys=action_keys,
                include_gripper=include_gripper,
                use_rot6d=action_rot6d,
            )
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"[OK] Converted episode {ep_idx + 1}/{len(episode_dirs)} with {len(frame_meta)} frames.")

    if args.copy_points:
        points_src = raw_root / "points"
        points_dst = out_root / "points"
        if points_src.exists():
            shutil.copytree(points_src, points_dst, dirs_exist_ok=True)
            print("[OK] Copied points/ directory.")

    print(f"[OK] LeRobot dataset saved to: {out_root}")


if __name__ == "__main__":
    main()
