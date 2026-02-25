#!/usr/bin/env python3
"""Common helpers for Isaac ROS2 simulation control."""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from isaac_ros2_messages.srv import GetPrimAttribute, SetPrimAttribute
from lerobot_robot_ros2.utils.pose_utils import (  # pyright: ignore[reportMissingImports]
    action_from_pose,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    rotate_vector_by_quat_inverse,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from simulation_interfaces.srv import GetEntityState, SetSimulationState

SERVICE_CALL_TIMEOUT = 8.0
SERVICE_CALL_RETRIES = 3
SERVICE_RETRY_DELAY = 0.2
T = TypeVar("T")

_service_node: Node | None = None
_service_lock = threading.Lock()


def _ensure_service_node() -> Node:
    global _service_node
    with _service_lock:
        if _service_node is not None:
            return _service_node
        if not rclpy.ok():
            rclpy.init()
        _service_node = rclpy.create_node(
            "lerobot_isaac_service_client",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        return _service_node


def _call_service_once(
    service_type: type,
    service_name: str,
    request: object,
    *,
    timeout: float = SERVICE_CALL_TIMEOUT,
) -> object:
    node = _ensure_service_node()
    client = node.create_client(service_type, service_name)
    try:
        if not client.wait_for_service(timeout_sec=max(0.1, timeout)):
            raise RuntimeError(f"Service '{service_name}' unavailable within {timeout:.1f}s")
        future = client.call_async(request)
        start = time.monotonic()
        while not future.done():
            if (time.monotonic() - start) > timeout:
                raise RuntimeError(f"Service '{service_name}' timed out after {timeout:.1f}s")
            rclpy.spin_once(node, timeout_sec=0.01)
        if future.exception() is not None:
            raise RuntimeError(f"Service '{service_name}' call failed: {future.exception()}")
        return future.result()
    finally:
        try:
            node.destroy_client(client)
        except Exception:
            pass


def _call_service_with_retry(
    call_fn: Callable[[], T],
    description: str,
    *,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> T:
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt_idx in range(attempts):
        try:
            return call_fn()
        except Exception as exc:
            last_error = exc
            if attempt_idx < (attempts - 1):
                time.sleep(max(0.0, retry_delay))
    raise RuntimeError(
        f"{description} failed after {attempts} attempts: {last_error or 'unknown error'}"
    )


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


def set_simulation_state(
    state: int,
    timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> None:
    request = SetSimulationState.Request()
    request.state.state = int(state)
    result = _call_service_with_retry(
        lambda: _call_service_once(
            SetSimulationState,
            "/set_simulation_state",
            request,
            timeout=timeout,
        ),
        f"set_simulation_state({state})",
        retries=retries,
        retry_delay=retry_delay,
    )
    if result.result.error_message:
        raise RuntimeError(
            f"set_simulation_state({state}) unsuccessful: {result.result.error_message}"
        )


def get_prim_translate_local(
    path: str,
    timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> tuple[float, float, float]:
    request = GetPrimAttribute.Request()
    request.path = path
    request.attribute = "xformOp:translate"
    result = _call_service_with_retry(
        lambda: _call_service_once(
            GetPrimAttribute,
            "/get_prim_attribute",
            request,
            timeout=timeout,
        ),
        f"get_prim_translate_local('{path}')",
        retries=retries,
        retry_delay=retry_delay,
    )
    if not result.success:
        raise RuntimeError(
            f"get_prim_attribute unsuccessful for '{path}': {result.message or 'unknown error'}"
        )
    vec = np.fromstring(result.value.strip().strip("[]"), sep=",")
    if vec.shape[0] != 3:
        raise RuntimeError(f"Invalid translate vector for '{path}': {result.value}")
    return float(vec[0]), float(vec[1]), float(vec[2])


def set_prim_translate_local(
    path: str,
    xyz: tuple[float, float, float],
    timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> None:
    x, y, z = xyz
    request = SetPrimAttribute.Request()
    request.path = path
    request.attribute = "xformOp:translate"
    request.value = f"[{x}, {y}, {z}]"
    result = _call_service_with_retry(
        lambda: _call_service_once(
            SetPrimAttribute,
            "/set_prim_attribute",
            request,
            timeout=timeout,
        ),
        f"set_prim_translate_local('{path}')",
        retries=retries,
        retry_delay=retry_delay,
    )
    if not result.success:
        raise RuntimeError(
            f"set_prim_attribute unsuccessful for '{path}': {result.message or 'unknown error'}"
        )


def randomize_object_xyz_after_reset(
    object_entity_path: str,
    enabled: bool = True,
    xyz_offset: tuple[float, float, float] | float = 0.04,
    timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> None:
    if not enabled:
        return
    if isinstance(xyz_offset, tuple):
        if len(xyz_offset) != 3:
            raise ValueError(f"xyz_offset must be a 3-tuple, got: {xyz_offset}")
        off_x, off_y, off_z = xyz_offset
    else:
        off_x, off_y, off_z = float(xyz_offset), float(xyz_offset), 0.0
    cur_x, cur_y, cur_z = get_prim_translate_local(
        object_entity_path,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    new_x = cur_x + random.uniform(-off_x, off_x)
    new_y = cur_y + random.uniform(-off_y, off_y)
    new_z = cur_z + random.uniform(-off_z, off_z)
    set_prim_translate_local(
        object_entity_path,
        (new_x, new_y, new_z),
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    print(
        f"[OK] Randomized object local xyz: "
        f"({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f}) -> ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})"
    )


def reset_simulation_and_randomize_object(
    object_entity_path: str,
    *,
    sim_state_reset: int = 0,
    sim_state_playing: int = 1,
    sim_service_timeout: float = SERVICE_CALL_TIMEOUT,
    enable_randomization: bool = True,
    xyz_offset: tuple[float, float, float] | float = 0.04,
    prim_attr_timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
    post_reset_wait: float = 1.0,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    set_simulation_state(
        sim_state_reset,
        timeout=sim_service_timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    set_simulation_state(
        sim_state_playing,
        timeout=sim_service_timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    print("[OK] Simulation reset completed")
    randomize_object_xyz_after_reset(
        object_entity_path,
        enabled=enable_randomization,
        xyz_offset=xyz_offset,
        timeout=prim_attr_timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    (sleep_fn or time.sleep)(post_reset_wait)


def get_entity_pose_world_service(
    entity: str,
    timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    request = GetEntityState.Request()
    request.entity = entity
    result = _call_service_with_retry(
        lambda: _call_service_once(
            GetEntityState,
            "/get_entity_state",
            request,
            timeout=timeout,
        ),
        f"get_entity_pose_world_service('{entity}')",
        retries=retries,
        retry_delay=retry_delay,
    )
    if result.result.error_message:
        raise RuntimeError(
            f"get_entity_state unsuccessful for '{entity}': {result.result.error_message}"
        )
    pose = result.state.pose
    return (
        (float(pose.position.x), float(pose.position.y), float(pose.position.z)),
        (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ),
    )


def get_object_pose_from_service(
    base_world_pos: tuple[float, float, float],
    base_world_quat: tuple[float, float, float, float],
    object_entity_path: str,
    *,
    include_orientation: bool = False,
    entity_state_timeout: float = SERVICE_CALL_TIMEOUT,
    retries: int = SERVICE_CALL_RETRIES,
    retry_delay: float = SERVICE_RETRY_DELAY,
) -> Pose:
    base_wx, base_wy, base_wz = base_world_pos
    base_q = base_world_quat
    (obj_wx, obj_wy, obj_wz), obj_q = get_entity_pose_world_service(
        object_entity_path,
        timeout=entity_state_timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    rel_world = (obj_wx - base_wx, obj_wy - base_wy, obj_wz - base_wz)
    rel_x, rel_y, rel_z = rotate_vector_by_quat_inverse(rel_world, base_q)

    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    if include_orientation:
        q_obj_in_base = quat_multiply(
            quat_conjugate(quat_normalize(base_q)),
            quat_normalize(obj_q),
        )
        qx, qy, qz, qw = quat_normalize(q_obj_in_base)

    pose = Pose()
    pose.position.x = rel_x
    pose.position.y = rel_y
    pose.position.z = rel_z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose
