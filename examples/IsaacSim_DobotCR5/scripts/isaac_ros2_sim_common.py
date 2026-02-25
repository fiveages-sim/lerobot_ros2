#!/usr/bin/env python3
"""Common helpers for Isaac ROS2 simulation control."""

from __future__ import annotations

import random
import re
import subprocess
import threading
import time
from typing import Callable

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter


class SimTimeHelper:
    """Provide ROS(sim) time utilities."""

    def __init__(self, poll_period: float = 0.01) -> None:
        self._poll_period = poll_period
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(
            "lerobot_sim_time_helper",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        use_sim_time = self._node.get_parameter("use_sim_time").value
        print(f"[Clock] use_sim_time={use_sim_time}")

    def now_seconds(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def sleep(self, duration: float) -> None:
        if duration <= 0.0:
            return
        start = self.now_seconds()
        while (self.now_seconds() - start) < duration:
            time.sleep(self._poll_period)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


def action_from_pose(pose: Pose, gripper: float) -> dict[str, float]:
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


def set_simulation_state(state: int, timeout: float = 8.0) -> None:
    request = f"{{state: {{state: {state}}}}}"
    command = [
        "ros2",
        "service",
        "call",
        "/set_simulation_state",
        "simulation_interfaces/srv/SetSimulationState",
        request,
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"set_simulation_state({state}) failed: {stderr or stdout or 'unknown error'}"
        )
    err_match = re.search(r"error_message='([^']*)'", result.stdout)
    if err_match and err_match.group(1):
        raise RuntimeError(
            f"set_simulation_state({state}) unsuccessful: {err_match.group(1)}"
        )


def get_prim_translate_local(path: str, timeout: float = 8.0) -> tuple[float, float, float]:
    request = f'{{path: "{path}", attribute: "xformOp:translate"}}'
    command = [
        "ros2",
        "service",
        "call",
        "/get_prim_attribute",
        "isaac_ros2_messages/srv/GetPrimAttribute",
        request,
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"get_prim_attribute failed for '{path}': {stderr or stdout or 'unknown error'}"
        )
    text = result.stdout
    if "success=True" not in text:
        raise RuntimeError(f"get_prim_attribute unsuccessful for '{path}': {text.strip()}")
    match = re.search(r"value='(\[[^']+\])'", text)
    if match is None:
        raise RuntimeError(f"Cannot parse xformOp:translate for '{path}': {text.strip()}")
    vec = np.fromstring(match.group(1).strip("[]"), sep=",")
    if vec.shape[0] != 3:
        raise RuntimeError(f"Invalid translate vector for '{path}': {match.group(1)}")
    return float(vec[0]), float(vec[1]), float(vec[2])


def set_prim_translate_local(
    path: str, xyz: tuple[float, float, float], timeout: float = 8.0
) -> None:
    x, y, z = xyz
    request = f'{{path: "{path}", attribute: "xformOp:translate", value: [{x}, {y}, {z}]}}'
    command = [
        "ros2",
        "service",
        "call",
        "/set_prim_attribute",
        "isaac_ros2_messages/srv/SetPrimAttribute",
        request,
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"set_prim_attribute failed for '{path}': {stderr or stdout or 'unknown error'}"
        )
    if "success=True" not in result.stdout:
        raise RuntimeError(
            f"set_prim_attribute unsuccessful for '{path}': {result.stdout.strip()}"
        )


def randomize_object_xy_after_reset(
    object_entity_path: str,
    enabled: bool = True,
    xy_offset: float = 0.04,
    timeout: float = 8.0,
) -> None:
    if not enabled:
        return
    cur_x, cur_y, cur_z = get_prim_translate_local(object_entity_path, timeout=timeout)
    new_x = cur_x + random.uniform(-xy_offset, xy_offset)
    new_y = cur_y + random.uniform(-xy_offset, xy_offset)
    set_prim_translate_local(object_entity_path, (new_x, new_y, cur_z), timeout=timeout)
    print(
        f"[OK] Randomized object local x/y: "
        f"({cur_x:.3f}, {cur_y:.3f}) -> ({new_x:.3f}, {new_y:.3f})"
    )


def reset_simulation_and_randomize_object(
    object_entity_path: str,
    *,
    sim_state_reset: int = 0,
    sim_state_playing: int = 1,
    sim_service_timeout: float = 8.0,
    enable_randomization: bool = True,
    xy_offset: float = 0.04,
    prim_attr_timeout: float = 8.0,
    post_reset_wait: float = 1.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    set_simulation_state(sim_state_reset, timeout=sim_service_timeout)
    set_simulation_state(sim_state_playing, timeout=sim_service_timeout)
    print("[OK] Simulation reset completed")
    randomize_object_xy_after_reset(
        object_entity_path,
        enabled=enable_randomization,
        xy_offset=xy_offset,
        timeout=prim_attr_timeout,
    )
    (sleep_fn or time.sleep)(post_reset_wait)


def get_entity_pose_world_service(
    entity: str, timeout: float = 8.0
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    request = f'{{entity: "{entity}"}}'
    command = [
        "ros2",
        "service",
        "call",
        "/get_entity_state",
        "simulation_interfaces/srv/GetEntityState",
        request,
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            f"get_entity_state failed for '{entity}': {stderr or stdout or 'unknown error'}"
        )
    text = result.stdout
    err_match = re.search(r"error_message='([^']*)'", text)
    if err_match and err_match.group(1):
        raise RuntimeError(
            f"get_entity_state unsuccessful for '{entity}': {err_match.group(1)}"
        )
    pos_match = re.search(
        r"position=geometry_msgs\.msg\.Point\(x=([-+0-9.eE]+), y=([-+0-9.eE]+), z=([-+0-9.eE]+)\)",
        text,
    )
    ori_match = re.search(
        r"orientation=geometry_msgs\.msg\.Quaternion\(x=([-+0-9.eE]+), y=([-+0-9.eE]+), z=([-+0-9.eE]+), w=([-+0-9.eE]+)\)",
        text,
    )
    if pos_match is None or ori_match is None:
        raise RuntimeError(f"Cannot parse pose from get_entity_state response: {text.strip()}")
    return (
        (float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))),
        (
            float(ori_match.group(1)),
            float(ori_match.group(2)),
            float(ori_match.group(3)),
            float(ori_match.group(4)),
        ),
    )


def _quat_multiply(
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


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    arr = np.array(q, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    arr /= n
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _rotate_vector_by_quat_inverse(
    vec: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    q_inv = _quat_conjugate(_quat_normalize(quat_xyzw))
    v_quat = (vec[0], vec[1], vec[2], 0.0)
    rotated = _quat_multiply(_quat_multiply(q_inv, v_quat), _quat_conjugate(q_inv))
    return (rotated[0], rotated[1], rotated[2])


def get_object_pose_from_service(
    base_world_pos: tuple[float, float, float],
    base_world_quat: tuple[float, float, float, float],
    object_entity_path: str,
    *,
    include_orientation: bool = False,
    entity_state_timeout: float = 8.0,
) -> Pose:
    base_wx, base_wy, base_wz = base_world_pos
    base_q = base_world_quat
    (obj_wx, obj_wy, obj_wz), obj_q = get_entity_pose_world_service(
        object_entity_path, timeout=entity_state_timeout
    )
    rel_world = (obj_wx - base_wx, obj_wy - base_wy, obj_wz - base_wz)
    rel_x, rel_y, rel_z = _rotate_vector_by_quat_inverse(rel_world, base_q)

    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    if include_orientation:
        q_obj_in_base = _quat_multiply(
            _quat_conjugate(_quat_normalize(base_q)),
            _quat_normalize(obj_q),
        )
        qx, qy, qz, qw = _quat_normalize(q_obj_in_base)

    pose = Pose()
    pose.position.x = rel_x
    pose.position.y = rel_y
    pose.position.z = rel_z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose
