"""Pose/action and quaternion helpers for ROS2 robots."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose


def action_from_pose(pose: Pose, gripper: float) -> dict[str, float]:
    """Build standard LeRobot ROS2 action dict from a pose."""
    return {
        "end_effector.position.x": pose.position.x,
        "end_effector.position.y": pose.position.y,
        "end_effector.position.z": pose.position.z,
        "end_effector.orientation.x": pose.orientation.x,
        "end_effector.orientation.y": pose.orientation.y,
        "end_effector.orientation.z": pose.orientation.z,
        "end_effector.orientation.w": pose.orientation.w,
        "gripper.position": float(np.clip(gripper, 0.0, 1.0)),
    }


def quat_multiply(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    arr = np.array(q, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    arr /= n
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def rotate_vector_by_quat_inverse(
    vec: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    q_inv = quat_conjugate(quat_normalize(quat_xyzw))
    v_quat = (vec[0], vec[1], vec[2], 0.0)
    rotated = quat_multiply(quat_multiply(q_inv, v_quat), quat_conjugate(q_inv))
    return (rotated[0], rotated[1], rotated[2])


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in xyzw order to rotation-6D."""
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 elements, got shape {quat.shape}")
    quat = np.atleast_2d(quat)

    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    quat = quat / norms

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)

    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)
    return np.atleast_2d(rot6d)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert rotation-6D vector(s) to quaternion(s) in xyzw order."""
    rot = np.asarray(rot6d, dtype=np.float32)
    if rot.shape[-1] != 6:
        raise ValueError(f"Expected rot6d with 6 elements, got shape {rot.shape}")
    rot = np.atleast_2d(rot)

    a1 = rot[:, 0:3]
    a2 = rot[:, 3:6]

    b1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1_norm = np.where(b1_norm < 1e-8, 1.0, b1_norm)
    b1 = a1 / b1_norm

    proj = np.sum(a2 * b1, axis=-1, keepdims=True) * b1
    b2_raw = a2 - proj
    b2_norm = np.linalg.norm(b2_raw, axis=-1, keepdims=True)
    b2_norm = np.where(b2_norm < 1e-8, 1.0, b2_norm)
    b2 = b2_raw / b2_norm

    b3 = np.cross(b1, b2, axis=-1)

    m00 = b1[:, 0]
    m01 = b2[:, 0]
    m02 = b3[:, 0]
    m10 = b1[:, 1]
    m11 = b2[:, 1]
    m12 = b3[:, 1]
    m20 = b1[:, 2]
    m21 = b2[:, 2]
    m22 = b3[:, 2]

    trace = m00 + m11 + m22
    quat = np.zeros((rot.shape[0], 4), dtype=np.float32)  # xyzw

    mask = trace > 0.0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        quat[mask, 3] = 0.25 * s
        quat[mask, 0] = (m21[mask] - m12[mask]) / s
        quat[mask, 1] = (m02[mask] - m20[mask]) / s
        quat[mask, 2] = (m10[mask] - m01[mask]) / s

    mask0 = (~mask) & (m00 > m11) & (m00 > m22)
    if np.any(mask0):
        s = np.sqrt(1.0 + m00[mask0] - m11[mask0] - m22[mask0]) * 2.0
        quat[mask0, 3] = (m21[mask0] - m12[mask0]) / s
        quat[mask0, 0] = 0.25 * s
        quat[mask0, 1] = (m01[mask0] + m10[mask0]) / s
        quat[mask0, 2] = (m02[mask0] + m20[mask0]) / s

    mask1 = (~mask) & (~mask0) & (m11 > m22)
    if np.any(mask1):
        s = np.sqrt(1.0 + m11[mask1] - m00[mask1] - m22[mask1]) * 2.0
        quat[mask1, 3] = (m02[mask1] - m20[mask1]) / s
        quat[mask1, 0] = (m01[mask1] + m10[mask1]) / s
        quat[mask1, 1] = 0.25 * s
        quat[mask1, 2] = (m12[mask1] + m21[mask1]) / s

    mask2 = (~mask) & (~mask0) & (~mask1)
    if np.any(mask2):
        s = np.sqrt(1.0 + m22[mask2] - m00[mask2] - m11[mask2]) * 2.0
        quat[mask2, 3] = (m10[mask2] - m01[mask2]) / s
        quat[mask2, 0] = (m02[mask2] + m20[mask2]) / s
        quat[mask2, 1] = (m12[mask2] + m21[mask2]) / s
        quat[mask2, 2] = 0.25 * s

    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    quat = quat / norms
    return np.atleast_2d(quat.astype(np.float32))
