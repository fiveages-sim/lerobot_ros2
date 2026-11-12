#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""末端轨迹采集 / 回放（EE-only，面向全身控制 OCS2）

为什么要这个工具
================
采集端最终只训练「双臂末端位姿 + 夹爪开合」，推理端也只输出这两样，再交给运控
(OCS2) 去做全身 IK/MPC。既然如此，验证数据是否可用就不需要碰 Isaac 视窗、
不需要回放关节角、更不需要管底盘开合——直接把采集到的末端轨迹喂给运控，
看运控能不能跟住，就是最贴近部署链路的验证。

坐标系
------
OCS2 在轮式底盘模式下，末端目标全部在 **world** 下
（`WbcCtrlComponent.h`: `ref_ctx.base_frame = hasMobileBase() ? "world" : minfo.baseFrame`），
`/left_current_pose`、`/right_current_pose` 的 header.frame_id 也是 world，
frame 是 `model_information.eeFrame`（= `left_tcp` / `right_tcp`）。

因此：
- 采集：直接存 `/left_current_pose` / `/right_current_pose` 的四元数位姿，天然就是
  运控的目标空间，**不需要任何 TF 换算**，也不受底盘漂移影响（world 是绝对系）。
- 回放：把同样的位姿原样发回 `/left_target` / `/right_target`（`Pose`，无 frame，
  控制器按 base_frame_=world 解释）。

话题
----
采集（订阅）
  /left_current_pose     geometry_msgs/PoseStamped  world 下 left_tcp 实测位姿 → observation
  /right_current_pose    同上
  /left_current_target   geometry_msgs/PoseStamped  运控当前目标位姿 → action
  /right_current_target  同上
  /joint_states          取 left_gripper_joint / right_gripper_joint 实际开合
  /left_gripper_controller/target_command   std_msgs/Int32  1=开 0=关（命令值）
  /right_gripper_controller/target_command  同上

回放（发布）
  /left_target          geometry_msgs/Pose         立即绝对目标（等价 VR 单帧，无插值）
  /right_target         geometry_msgs/Pose
  /left_gripper_controller/target_command   std_msgs/Int32
  /right_gripper_controller/target_command  std_msgs/Int32
  /dual_target/stamped  geometry_msgs/Path（2 点：[left, right]）仅用于「平滑走到起点」

实测 vs 指令
------------
- **实测**（`*_current_pose`）是控制器跟出来的结果 = 训练时的 observation。
  它已经包含跟踪误差，拿它当 action 会把误差学进去。
- **指令**（`*_current_target`）是当时真正下发的目标 = 训练的 action 标签，
  也是回放该复现的东西。
- 文件里两套都存；`inspect` 会打印二者的差值。如果差值很大，说明采集时
  运控就跟不住，那段轨迹不适合直接用于训练。

用法
----
    # 0) 先看机器人/话题是否就绪
    ./ee_traj.sh check

    # 1) 采集一段（编排在跑，本脚本只旁听）
    ./ee_traj.sh record -o ~/ee_traj/pick_01.npz

    # 2) 只读检查（不需要 ROS）
    ./ee_traj.sh inspect ~/ee_traj/pick_01.npz

    # 3) 截取子段（比如只验证「交接」那 3 秒）
    ./ee_traj.sh slice ~/ee_traj/pick_01.npz --start-sec 12.0 --end-sec 15.0

    # 4) 回放给运控验证
    ./ee_traj.sh replay ~/ee_traj/pick_01.npz --approach-duration 3.0

注意事项
--------
- 双臂耦合（BIMANUAL_COUPLED）时 `/right_target` 会被忽略（左臂 leader 优先），
  所以回放前脚本会先发 `ARMS_INDEPENDENT`。
- 单臂被 disable 时该臂的 EE 命令会被控制器丢弃（`block_*_external_ee_`），
  `check` 会提前报出来。`LEFT_ARM_DISABLE` 期间左手的指令是「保持」，
  回放时左手会被拉到那段保持位姿——需要时用 `slice` 掐掉。
- `replay` 结束时会打印「指令 vs 实测」的跟踪误差——这才是本次验证的核心结论。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TOPIC_LEFT_CUR = "/left_current_pose"
TOPIC_RIGHT_CUR = "/right_current_pose"
TOPIC_LEFT_CUR_TGT = "/left_current_target"
TOPIC_RIGHT_CUR_TGT = "/right_current_target"
TOPIC_LEFT_TGT = "/left_target"
TOPIC_RIGHT_TGT = "/right_target"
TOPIC_DUAL_TGT_STAMPED = "/dual_target/stamped"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_LEFT_GRIP_CMD = "/left_gripper_controller/target_command"
TOPIC_RIGHT_GRIP_CMD = "/right_gripper_controller/target_command"
TOPIC_WBC_STATE = "/ocs2_wbc_controller/current_state"
TOPIC_MODE_COMMAND = "/mode_command"

GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0
GRIPPER_UNKNOWN = -1

LEFT_GRIP_JOINT = "left_gripper_joint"
RIGHT_GRIP_JOINT = "right_gripper_joint"

# Dex1 夹爪关节行程（来自 adaptive_gripper_controller 的 joint limits）。
# 用途：把 /joint_states 里的夹爪实测位置换算成 0/1 开合。
# 见 `grip_from_position` 的注释 —— 为什么不用 target_command 话题。
GRIPPER_CLOSED_POS = -0.020
GRIPPER_OPEN_POS = 0.0245

# 全身运控可优化关节（full_body 控制器 joints 列表顺序，来自 ros2_controllers.yaml）
# 采集时要按名字取，不依赖顺序。
#
# 末尾两个是 Dex1 夹爪关节：它们不在运控的优化集合里（运控不管夹爪），
# 但层2 回放时必须一起驱动，否则渲染出来的夹爪永远停在初始开合状态，
# 手里的试管会与夹爪脱开。
ARM_JOINTS = [
    "LZ_mt_Joint", "LZ_it_Joint", "waist_pitch_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # 夹爪关节（Dex1）。运控不优化它们，但层2 回放必须驱动，否则手里物体与夹爪脱开。
    #
    # 关键：Dex1 每只手有**两个**独立的 prismatic 关节（joint 和 joint1），
    # 分别驱动两个手指。只驱动 joint 会导致一指闭合、另一指不动，
    # 渲染出来看起来仍像「张开」。两个都要录。
    "left_gripper_joint", "left_gripper_joint1",
    "right_gripper_joint", "right_gripper_joint1",
]

DEFAULT_FRAME = "world"

#: 与 fasim.replicator.sdg.episode.SCHEMA_VERSION 保持一致。
#: 层2（Replicator 渲染）只认这个版本，改字段名时要同步两处。
EPISODE_SCHEMA_VERSION = 2

# 备注：轮式底盘模式下 OCS2 的 ref_ctx.base_frame 在源码里写的是 "world"，
# 但实测 /ocs2_wbc_controller 的 base_frame 参数是 "base_footprint"
# （world_frame 才是 "world"）。因此**不能硬编码 frame**：
#   - 采集时如实记录 /left_current_pose 的 header.frame_id
#   - 回放时用同一个 frame 回发，控制器内部按自己的 base_frame 处理
# 两者一致时无需任何 TF 换算；不一致时由控制器自己做 TF。

# 浮点容差：判断夹爪命令是否变化
EPS = 1e-6


# ---------------------------------------------------------------------------
# ROS 环境引导
# ---------------------------------------------------------------------------

ROS_ENV_HINT = """\
无法导入 rclpy。请在已 source 的 ROS 2 环境里运行，例如：

    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export ZENOH_SESSION_CONFIG_URI=/home/fiveages/fa_w2_ws/zenoh/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5

Isaac 侧用的是 newton kit + zenoh，缺 RMW/ZENOH 配置会看不到话题。
"""


def _import_ros():
    try:
        import rclpy
        from geometry_msgs.msg import Pose, PoseStamped
        from nav_msgs.msg import Path
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Int32, String
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(ROS_ENV_HINT, file=sys.stderr)
        raise SystemExit(2)
    return {
        "rclpy": rclpy,
        "Pose": Pose,
        "PoseStamped": PoseStamped,
        "Path": Path,
        "JointState": JointState,
        "Int32": Int32,
        "String": String,
    }


# ---------------------------------------------------------------------------
# 时钟：优先 ROS 时间（Isaac 用 /clock），否则退回墙钟
# ---------------------------------------------------------------------------

class Pacer:
    """按 ROS 时间推进，缺失 /clock 时自动退回 monotonic。

    不依赖 rclpy 的 use_sim_time 参数：rclpy 节点默认 use_sim_time=False，
    此时 node.get_clock().now() 返回的是系统墙钟（非 0），无法据此判断
    /clock 是否存在。这里直接订阅 /clock，收到消息才算 ROS 时间可用。

    ``auto`` 模式下**延迟决定**：构造时不判断，首次调用 :meth:`now` 时若已有
    /clock 就走 ROS 时间，否则走墙钟。这样「先启动脚本、后按 Play」也能正确
    切到仿真时间（构造时仿真还没 Play，2 秒等待必然超时）。
    """

    def __init__(self, node, use_sim_time: str = "auto") -> None:
        self._node = node
        self._lock = threading.Lock()
        self._sim_now: float | None = None
        self._fallback = use_sim_time == "wall"
        self._auto = use_sim_time == "auto"
        self._decided = not self._auto
        self._sub = None

        if not self._fallback:
            try:
                from rosgraph_msgs.msg import Clock
                self._sub = node.create_subscription(
                    Clock, "/clock", self._on_clock, 10)
            except Exception:  # noqa: BLE001
                self._fallback = True
                self._decided = True
                return

            if use_sim_time == "ros":
                # 显式指定 ros：阻塞等 /clock（最多 8s）
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline and self._sim_now is None:
                    time.sleep(0.02)
                if self._sim_now is None:
                    print("[warn] 指定了 --use-sim-time ros，但 /clock 无消息；"
                          "仿真是否在 Play？")
                    self._fallback = True

    def _decide(self) -> None:
        """首次取时间时决定走 ROS 时间还是墙钟。"""
        if self._decided:
            return
        self._decided = True
        if self._sim_now is None:
            self._fallback = True

    def _on_clock(self, msg) -> None:
        with self._lock:
            self._sim_now = msg.clock.sec + msg.clock.nanosec * 1e-9

    @property
    def using_ros_time(self) -> bool:
        return not self._fallback

    def now(self) -> float:
        if not self._decided:
            self._decide()
        if self._fallback:
            return time.monotonic()
        with self._lock:
            return self._sim_now if self._sim_now is not None else 0.0

    def sleep_until(self, target: float) -> None:
        """睡到目标时刻；ROS 时间可能被暂停，所以带墙钟兜底。"""
        wall_deadline = time.monotonic() + 30.0
        while True:
            remaining = target - self.now()
            if remaining <= 0:
                return
            if time.monotonic() > wall_deadline:
                return  # 时间源卡住（比如暂停），不死等
            time.sleep(min(remaining, 0.005))

    def sleep(self, seconds: float) -> None:
        self.sleep_until(self.now() + seconds)


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class Traj:
    """一次采集/回放用的末端轨迹。

    同时保存两套位姿：
    - ``*_pos`` / ``*_quat``      实测（/left_current_pose），= observation
    - ``*_cmd_pos`` / ``*_cmd_quat`` 指令目标（/left_current_target），= action

    训练时 observation 用实测、action 用指令；回放验证时用指令（复现意图，
    不含控制器滞后）。若目标话题没采到，``cmd_*`` 退化为实测。
    """

    t: np.ndarray                      # (N,) 采样时刻（ROS 秒）
    left_pos: np.ndarray               # (N,3) 实测
    left_quat: np.ndarray              # (N,4) xyzw 实测
    right_pos: np.ndarray
    right_quat: np.ndarray
    left_grip: np.ndarray              # (N,) 0/1/-1 指令
    right_grip: np.ndarray
    left_grip_pos: np.ndarray          # (N,) 关节实测，NaN 表示未知
    right_grip_pos: np.ndarray
    frame_id: str = DEFAULT_FRAME
    meta: dict[str, Any] = field(default_factory=dict)
    left_cmd_pos: np.ndarray | None = None
    left_cmd_quat: np.ndarray | None = None
    right_cmd_pos: np.ndarray | None = None
    right_cmd_quat: np.ndarray | None = None
    # 关节角：ACT 类方法要复刻完整任务流程，改 .info 权重时末端会变而关节角不变，
    # 所以关节角必须单独存一份。
    joint_pos: np.ndarray | None = None       # (N, J) 实测关节角
    joint_vel: np.ndarray | None = None       # (N, J) 实测关节速度
    joint_names: list[str] | None = None      # (J,) 与上面两列对应
    # 交互物体位姿：label -> (pos(N,3), quat(N,4))
    # 试管等是独立刚体，不受关节角驱动。层2 回放必须显式设定它们的世界位姿，
    # 否则物体停在初始位置、机器人手却做抓取动作，画面完全错位。
    objects: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def __len__(self) -> int:
        return int(self.t.shape[0])

    # ---- 指令位姿（缺省退化到实测）----

    def has_cmd(self) -> bool:
        return self.left_cmd_pos is not None and self.right_cmd_pos is not None

    def action(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """返回回放/训练用的 action 位姿（优先指令）。"""
        if side == "left":
            if self.left_cmd_pos is not None:
                return self.left_cmd_pos, self.left_cmd_quat
            return self.left_pos, self.left_quat
        if self.right_cmd_pos is not None:
            return self.right_cmd_pos, self.right_cmd_quat
        return self.right_pos, self.right_quat

    # ---- 存取 ----

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            t=self.t,
            left_pos=self.left_pos,
            left_quat=self.left_quat,
            right_pos=self.right_pos,
            right_quat=self.right_quat,
            left_grip=self.left_grip,
            right_grip=self.right_grip,
            left_grip_pos=self.left_grip_pos,
            right_grip_pos=self.right_grip_pos,
            frame_id=np.array(self.frame_id),
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
        )
        if self.has_cmd():
            payload.update(
                left_cmd_pos=self.left_cmd_pos,
                left_cmd_quat=self.left_cmd_quat,
                right_cmd_pos=self.right_cmd_pos,
                right_cmd_quat=self.right_cmd_quat,
            )
        if self.joint_pos is not None:
            payload.update(
                joint_pos=self.joint_pos,
                joint_vel=self.joint_vel if self.joint_vel is not None else np.zeros_like(self.joint_pos),
                joint_names=np.array(self.joint_names or []),
            )
        # 与 fasim.replicator.sdg.episode 的 schema 对齐：
        # 层2 的 render 命令直接吃这个文件，字段名必须一致。
        payload["schema_version"] = np.array(EPISODE_SCHEMA_VERSION)
        # 交互物体位姿：按 obj/<label>_pos / obj/<label>_quat 写盘
        if self.objects:
            for label, (pos, quat) in self.objects.items():
                payload[f"obj/{label}_pos"] = pos
                payload[f"obj/{label}_quat"] = quat
        np.savez_compressed(p, **payload)
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Traj":
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"[ERROR] 找不到文件: {p}")
        with np.load(p, allow_pickle=False) as z:
            meta_raw = z["meta"].item() if "meta" in z.files else "{}"
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            n = len(z["t"])
            return cls(
                t=z["t"],
                left_pos=z["left_pos"],
                left_quat=z["left_quat"],
                right_pos=z["right_pos"],
                right_quat=z["right_quat"],
                left_grip=z["left_grip"] if "left_grip" in z.files else np.full(n, GRIPPER_UNKNOWN, float),
                right_grip=z["right_grip"] if "right_grip" in z.files else np.full(n, GRIPPER_UNKNOWN, float),
                left_grip_pos=z["left_grip_pos"] if "left_grip_pos" in z.files else np.full(n, np.nan),
                right_grip_pos=z["right_grip_pos"] if "right_grip_pos" in z.files else np.full(n, np.nan),
                frame_id=z["frame_id"].item() if "frame_id" in z.files else DEFAULT_FRAME,
                meta=json.loads(str(meta_raw)),
                left_cmd_pos=z["left_cmd_pos"] if "left_cmd_pos" in z.files else None,
                left_cmd_quat=z["left_cmd_quat"] if "left_cmd_quat" in z.files else None,
                right_cmd_pos=z["right_cmd_pos"] if "right_cmd_pos" in z.files else None,
                right_cmd_quat=z["right_cmd_quat"] if "right_cmd_quat" in z.files else None,
                joint_pos=z["joint_pos"] if "joint_pos" in z.files else None,
                joint_vel=z["joint_vel"] if "joint_vel" in z.files else None,
                joint_names=(list(z["joint_names"]) if "joint_names" in z.files else None),
            )

    def slice(self, start: int = 0, end: int | None = None) -> "Traj":
        e = len(self) if end is None else min(end, len(self))
        s = max(0, start)

        def cut(a):
            return None if a is None else a[s:e]

        return Traj(
            t=self.t[s:e],
            left_pos=self.left_pos[s:e],
            left_quat=self.left_quat[s:e],
            right_pos=self.right_pos[s:e],
            right_quat=self.right_quat[s:e],
            left_grip=self.left_grip[s:e],
            right_grip=self.right_grip[s:e],
            left_grip_pos=self.left_grip_pos[s:e],
            right_grip_pos=self.right_grip_pos[s:e],
            frame_id=self.frame_id,
            meta=self.meta,
            left_cmd_pos=cut(self.left_cmd_pos),
            left_cmd_quat=cut(self.left_cmd_quat),
            right_cmd_pos=cut(self.right_cmd_pos),
            right_cmd_quat=cut(self.right_cmd_quat),
            joint_pos=cut(self.joint_pos),
            joint_vel=cut(self.joint_vel),
            joint_names=self.joint_names,
        )

    def summary(self) -> str:
        n = len(self)
        if n == 0:
            return "空轨迹"
        dt = np.diff(self.t) if n > 1 else np.array([0.0])
        dur = float(self.t[-1] - self.t[0])
        lp, rp = self.left_pos, self.right_pos
        lines = [
            f"帧数          : {n}",
            f"时长          : {dur:.2f} s",
            f"帧间隔 median : {float(np.median(dt)) * 1e3:.1f} ms"
            f"  (min {float(dt.min()) * 1e3:.1f} / max {float(dt.max()) * 1e3:.1f})",
            f"坐标系        : {self.frame_id}",
            f"左手 实测行程 : {_path_len(lp) * 100:.1f} cm"
            f"  范围 x[{lp[:, 0].min():.3f},{lp[:, 0].max():.3f}]"
            f" y[{lp[:, 1].min():.3f},{lp[:, 1].max():.3f}]"
            f" z[{lp[:, 2].min():.3f},{lp[:, 2].max():.3f}]",
            f"右手 实测行程 : {_path_len(rp) * 100:.1f} cm"
            f"  范围 x[{rp[:, 0].min():.3f},{rp[:, 0].max():.3f}]"
            f" y[{rp[:, 1].min():.3f},{rp[:, 1].max():.3f}]"
            f" z[{rp[:, 2].min():.3f},{rp[:, 2].max():.3f}]",
            f"左夹爪命令    : {_grip_summary(self.left_grip)}",
            f"右夹爪命令    : {_grip_summary(self.right_grip)}",
            f"左右间距      : 起 {np.linalg.norm(lp[0] - rp[0]) * 100:.1f} cm"
            f" → 末 {np.linalg.norm(lp[-1] - rp[-1]) * 100:.1f} cm",
        ]
        if self.has_cmd():
            lines.append(f"指令 vs 实测  : {self.cmd_gap()}")
            lines.append("回放数据源    : 指令目标（action）；实测仅作 observation")
        else:
            lines.append("回放数据源    : 实测位姿（未采到 *_current_target）")
        if self.joint_pos is not None:
            lines.append(f"关节角        : {self.joint_pos.shape[1]} 个 × {self.joint_pos.shape[0]} 帧（供 ACT 复刻）")
            lines.append(f"  顺序: {', '.join(self.joint_names or [])}")
        return "\n".join(lines)

    def cmd_gap(self) -> str:
        parts = []
        for side in ("left", "right"):
            cpos, cquat = self.action(side)
            mpos, mquat = ((self.left_pos, self.left_quat) if side == "left"
                           else (self.right_pos, self.right_quat))
            ep = np.linalg.norm(cpos - mpos, axis=1)
            eq = _quat_angle(cquat, mquat)
            parts.append(f"{'左' if side == 'left' else '右'} 位置 "
                         f"mean {ep.mean() * 1000:.1f} / max {ep.max() * 1000:.1f} mm，"
                         f"姿态 max {np.degrees(eq.max()):.1f}°")
        return "; ".join(parts)


def _path_len(pos: np.ndarray) -> float:
    if pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _grip_summary(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "无数据"
    known = arr[arr >= 0]
    if known.size == 0:
        return "未采集到命令（仅实测位置）"
    changes = int(np.sum(np.abs(np.diff(known)) > 0.5))
    n_open = int(np.sum(known > 0.5))
    return f"开 {n_open} 帧 / 关 {known.size - n_open} 帧，切换 {changes} 次"


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------

def _shutdown(rclpy, executor, node, spin_thread: threading.Thread | None = None) -> None:
    """有序关闭：先停 spin 线程，再销毁节点，最后 shutdown。

    顺序错了会触发 `terminate called without an active exception`
    （executor 还在 spin 时就 destroy_node / shutdown）。
    """
    try:
        executor.shutdown()
    except Exception:  # noqa: BLE001
        pass
    if spin_thread is not None and spin_thread.is_alive():
        spin_thread.join(timeout=2.0)
    try:
        node.destroy_node()
    except Exception:  # noqa: BLE001
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass


class _Latest:
    """线程安全的「最新值」缓存。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v: Any = None
        self.count = 0

    def set(self, v: Any) -> None:
        with self._lock:
            self._v = v
            self.count += 1

    def get(self) -> Any:
        with self._lock:
            return self._v


def _pose6_of(msg) -> tuple[list[float], list[float]]:
    """PoseStamped → ([x,y,z], [qx,qy,qz,qw])。"""
    p = msg.pose.position
    o = msg.pose.orientation
    return ([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])


def valid_target_pose(pos: list[float], quat: list[float]) -> bool:
    """判断一个 target 位姿是否有效。

    运控在「臂 disable / 进 HOLD / 重置 target 缓存」时会短暂把缓存清零并
    发布出去，表现为 position=(0,0,0) 且 orientation=(0,0,0,0)。
    这类帧不是真实动作，必须丢弃，否则训练会学到「跳回世界原点」。

    实测：编排录一遍 1752 帧里，左右各有 16 帧是这种全零占位。
    """
    if any(v != v for v in pos) or any(v != v for v in quat):
        return False
    qn = (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2) ** 0.5
    if qn < 1e-6:
        return False
    if abs(pos[0]) < 1e-9 and abs(pos[1]) < 1e-9 and abs(pos[2]) < 1e-9:
        return False
    return True


def grip_from_position(pos: float) -> float:
    """把夹爪关节的实测位置换算成 0/1 开合状态。

    为什么不用 ``/left_gripper_controller/target_command``：
    那个话题的 publisher count 是 0 —— ``adaptive_gripper_controller`` 只在
    收到外部命令时才转发，旁听者永远收不到（实测 1752 帧全为 -1）。
    而夹爪关节位置在 ``/joint_states`` 里一直都有，按行程阈值判定更可靠。
    """
    if pos != pos:  # NaN
        return float(GRIPPER_UNKNOWN)
    span = GRIPPER_OPEN_POS - GRIPPER_CLOSED_POS
    if abs(span) < 1e-9:
        return float(GRIPPER_UNKNOWN)
    norm = (pos - GRIPPER_CLOSED_POS) / span
    return 1.0 if norm > 0.5 else 0.0


# ---------------------------------------------------------------------------
# 交互物体位姿
# ---------------------------------------------------------------------------

#: 默认跟踪的交互物体（prim 路径）。
#:
#: 为什么要录：这些是**独立刚体**，不受关节角驱动。层2 回放时如果不显式设定
#: 它们的位姿，物体会停在初始位置，而机器人手在做抓取/插入动作 —— 画面完全错位。
#:
#: 只录**会被机器人移动**的刚体：
#:   test_tube         被抓起、交接、插入的试管
#:   test_tube_rack    插入目标架（hole_3 所在，被夹爪下探时可能推动）
#:   test_tube_rack_01 第二个架子（右侧，可能被碰撞）
#:
#: 不录 lab_table：桌面是静态网格（无 RigidBodyAPI），本就恒定不动。
#: 硬录它只会浪费每帧一次服务调用（实测 4 个物体把采样从 50Hz 拖到 15Hz）。
#: 需要时用 --track-prims 显式加上。
#:
#: 新增物体时同步更新 configs/unitree_g1d/tasks/sinopec_test_tube.yaml 的
#: tracked_prims（label 必须一致），否则 render 会报「没有对应 prim」并跳过。
DEFAULT_TRACK_PRIMS = [
    "/World/room/scene/test_tube",
    "/World/room/scene/test_tube_rack",
    "/World/room/scene/test_tube_rack_01",
]


def prim_label(prim_path: str) -> str:
    """把 prim 路径转成可做 key 的短名：/World/room/scene/test_tube → test_tube。"""
    return str(prim_path).rstrip("/").split("/")[-1]


class EntityPosePoller:
    """异步批量查询多个 prim 的位姿，不阻塞采样循环。

    为什么要异步：录制是 50Hz 循环，同步等 /get_entity_state 会严重拖慢采样
    （实测 4 个物体时从 15ms/帧 掉到 65ms/帧，正好 4 倍）。
    改成「发请求 → 下个周期取结果」后，采样循环只做一次非阻塞轮询。

    用法::

        poller = EntityPosePoller(node, prim_paths)
        ...
        while sampling:
            latest = poller.tick()   # {label: (pos, quat)} 非阻塞
            # 用 latest 落帧；返回的是最近一次成功的结果
    """

    def __init__(self, node, prim_paths: list[str], stale_sec: float = 1.0) -> None:
        self._paths = list(prim_paths)
        self._labels = [prim_label(p) for p in self._paths]
        self._client = None
        self._futures: dict[str, Any] = {}
        self._latest: dict[str, tuple[list[float], list[float]]] = {}
        self._stale_sec = stale_sec
        self._last_ok: dict[str, float] = {}
        self._fail_count: dict[str, int] = {lab: 0 for lab in self._labels}

        try:
            from simulation_interfaces.srv import GetEntityState
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] simulation_interfaces 不可用: {exc}")
            return
        self._srv_type = GetEntityState
        self._client = node.create_client(GetEntityState, "/get_entity_state")
        if not self._client.wait_for_service(timeout_sec=3.0):
            print("[warn] /get_entity_state 不可用（仿真未 Play？）")
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def tick(self) -> dict[str, tuple[list[float], list[float]]]:
        """推进一轮：取回已完成的请求，为空的槽位发新请求。立即返回。"""
        if self._client is None:
            return dict(self._latest)

        # 1) 收结果
        for label in list(self._futures):
            fut = self._futures.get(label)
            if fut is None or not fut.done():
                continue
            self._futures.pop(label, None)
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                self._fail_count[label] += 1
                continue
            if res is None or res.result.error_message:
                self._fail_count[label] += 1
                continue
            p = res.state.pose.position
            o = res.state.pose.orientation
            self._latest[label] = (
                [p.x, p.y, p.z],
                [o.x, o.y, o.z, o.w],
            )
            self._last_ok[label] = time.monotonic()
            self._fail_count[label] = 0

        # 2) 给空闲的物体发新请求（不等待）
        if self._client.service_is_ready():
            for label, path in zip(self._labels, self._paths):
                if label in self._futures:
                    continue
                try:
                    req = self._srv_type.Request()
                    req.entity = path
                    self._futures[label] = self._client.call_async(req)
                except Exception:  # noqa: BLE001
                    self._fail_count[label] += 1

        return dict(self._latest)

    def failure_summary(self) -> str:
        bad = {k: v for k, v in self._fail_count.items() if v > 0}
        if not bad:
            return ""
        return f"查询失败次数: {bad}"


def cmd_record(args: argparse.Namespace) -> int:
    ros = _import_ros()
    rclpy = ros["rclpy"]
    PoseStamped = ros["PoseStamped"]
    JointState = ros["JointState"]
    Int32 = ros["Int32"]

    rclpy.init()
    node = rclpy.create_node("ee_traj_record")

    latest: dict[str, _Latest] = {k: _Latest() for k in
                                  ("lp", "rp", "lgc", "rgc", "js", "lct", "rct")}

    def on_left(msg: PoseStamped) -> None:
        latest["lp"].set(msg)

    def on_right(msg: PoseStamped) -> None:
        latest["rp"].set(msg)

    def on_left_cmd(msg: PoseStamped) -> None:
        latest["lct"].set(msg)

    def on_right_cmd(msg: PoseStamped) -> None:
        latest["rct"].set(msg)

    def on_lgc(msg: Int32) -> None:
        latest["lgc"].set(int(msg.data))

    def on_rgc(msg: Int32) -> None:
        latest["rgc"].set(int(msg.data))

    def on_js(msg: JointState) -> None:
        latest["js"].set(msg)

    subs = [
        node.create_subscription(PoseStamped, args.left_pose_topic, on_left, 10),
        node.create_subscription(PoseStamped, args.right_pose_topic, on_right, 10),
        node.create_subscription(Int32, args.left_gripper_topic, on_lgc, 10),
        node.create_subscription(Int32, args.right_gripper_topic, on_rgc, 10),
        node.create_subscription(JointState, TOPIC_JOINT_STATES, on_js, 10),
        # 指令目标 = 训练用的 action（实测只是控制器跟出来的结果）
        node.create_subscription(PoseStamped, TOPIC_LEFT_CUR_TGT, on_left_cmd, 10),
        node.create_subscription(PoseStamped, TOPIC_RIGHT_CUR_TGT, on_right_cmd, 10),
    ]

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.1)

    # Pacer 需要 /clock，必须在 spin 启动之后构造
    pace = Pacer(node, args.use_sim_time)
    print(f"[info] 时间源: {'ROS(/clock)' if pace.using_ros_time else '墙钟(monotonic)'}")

    # ---- 交互物体位姿 ----
    # 试管/架子是独立刚体，不受关节角影响。不录它们的话层2 回放时
    # 物体停在初始位置，而机器人手在抓取 —— 画面完全错位。
    track_prims: list[str] = []
    if args.track_prims != "none":
        raw = (
            [s.strip() for s in args.track_prims.split(",") if s.strip()]
            if args.track_prims
            else list(DEFAULT_TRACK_PRIMS)
        )
        track_prims = raw

    # 异步轮询器：每帧发一批请求、取上一批结果，不阻塞采样。
    # 之前用同步查询（每帧等 4 次服务）把 50Hz 拖到 15Hz。
    poller = EntityPosePoller(node, track_prims) if track_prims else None
    dynamic_labels: set[str] = set()
    if track_prims:
        if poller is None or not poller.available:
            print("[warn] 物体位姿采集已跳过（服务不可用）")
            track_prims = []
            poller = None
        else:
            print(f"[info] 跟踪 {len(track_prims)} 个物体（异步查询，不阻塞采样）:")
            for p_ in track_prims:
                print(f"       {prim_label(p_):20s} {p_}")
            # 预热：先跑几轮让首帧就有数据
            for _ in range(8):
                poller.tick()
                time.sleep(0.01)
            dynamic_labels = set(poller.tick().keys())
    # 每个物体的最新位姿（查不到时沿用上一帧）
    obj_latest: dict[str, tuple[list[float], list[float]]] = {}

    # ---- 等数据就绪 ----
    # ---- 等待仿真启动 ----
    #
    # 为什么需要这个：
    # 仿真刚 Play 时，试管等物体会先做一次物理初始化沉降（几厘米），
    # 然后才稳定。之前脚本是「等到 /left_current_pose 有数据才开始采样」，
    # 而那时沉降早已结束 —— 录制数据里永远缺这一段。
    #
    # 现在的策略：只要 /clock 出现（= 仿真已 Play）就立刻开始采样。
    # 末端位姿话题若尚未就绪，用 NaN 占位，等它出现后自动补齐。
    # 这样「先启动本脚本 → 再按 Play」就能录到物体初始沉降。
    print(f"[info] 等待仿真启动（最多 {args.wait_timeout:.0f}s）...")
    print("       用法：先运行本脚本，再在 Isaac 里按 Play。")

    deadline = time.monotonic() + args.wait_timeout
    while time.monotonic() < deadline:
        if pace.now() > 0.0:
            break
        time.sleep(0.02)
    else:
        print("[ERROR] /clock 始终没有数据 —— 仿真是否在 Play？", file=sys.stderr)
        _shutdown(rclpy, executor, node, spin_thread)
        return 1

    sim_t0 = pace.now()
    print(f"[info] 仿真已启动 (t={sim_t0:.3f}s)")

    # ---- 关键：仿真已 Play，但运控可能还要几秒才发布末端位姿 ----
    #
    # 这段等待期里物体（试管）已经在做物理沉降，而机器人还没数据。
    # 如果直接开始采样，物体轨迹会被 NaN 污染的帧覆盖；
    # 如果等末端位姿，又会丢掉物体这段变化。
    #
    # 做法：**等待期也记录**，但只记物体 + 时间，机器人通道留 NaN。
    # 等末端位姿就绪后继续正常采样，整套数组长度一致、时间轴连续。
    pre_obj: list[dict] = []
    pre_t: list[float] = []
    print(f"[info] 等待运控就绪（末端位姿话题），期间持续记录物体位姿...")
    pose_deadline = time.monotonic() + max(10.0, args.wait_timeout)
    while time.monotonic() < pose_deadline:
        t_now = pace.now() - sim_t0
        snap = poller.tick() if poller is not None else {}
        obj_latest.update(snap)
        if snap or obj_latest:
            pre_t.append(t_now)
            pre_obj.append(dict(obj_latest))
        if latest["lp"].get() is not None and latest["rp"].get() is not None:
            break
        time.sleep(0.02)

    ready = latest["lp"].get() is not None and latest["rp"].get() is not None
    if ready:
        print(f"[info] 运控已就绪 (t={pace.now() - sim_t0:.3f}s)，"
              f"期间已记录物体 {len(pre_t)} 帧")
    else:
        print("[warn] 运控仍未就绪；继续采样，机器人通道会带 NaN")

    frame_id = (latest["lp"].get().header.frame_id if latest["lp"].get() is not None
                else DEFAULT_FRAME)
    print(f"[info] 末端位姿 frame_id = '{frame_id}'")

    # ---- 采集循环 ----
    # 用定时采样（固定 hz），而不是逐消息记录：回放需要均匀时间轴，
    # 且末端位姿与夹爪来自不同话题，统一网格才好对齐。
    hz = max(1.0, float(args.hz))
    period = 1.0 / hz
    print(f"[info] 以 {hz:.0f} Hz 采样，按 Ctrl+C 结束"
          f"{f'（{args.duration:.1f}s 后自动结束）' if args.duration > 0 else ''}")

    rec: dict[str, list] = {k: [] for k in
                            ("t", "lp", "lq", "rp", "rq", "lgc", "rgc", "lgp", "rgp",
                             "lcp", "lcq", "rcp", "rcq", "jp", "jv")}
    # 每个物体一组 (pos, quat) 序列
    obj_rec: dict[str, dict[str, list]] = {
        prim_label(p_): {"pos": [], "quat": []} for p_ in track_prims
    }
    t_start = pace.now()
    next_t = t_start
    interrupted = False

    # 时间基准统一用 sim_t0（仿真启动时刻），
    # 这样「等待运控期间」的物体帧和之后的正常帧在同一个时间轴上。
    t_origin = sim_t0

    # 把「等待运控期间」记录的物体帧先灌入
    if pre_t:
        print(f"[info] 并入等待期的 {len(pre_t)} 帧物体记录")
        nan3, nan4 = [float("nan")] * 3, [float("nan")] * 4
        for k, t_off in enumerate(pre_t):
            rec["t"].append(t_off)
            rec["lp"].append(nan3)
            rec["lq"].append(nan4)
            rec["rp"].append(nan3)
            rec["rq"].append(nan4)
            rec["lcp"].append(nan3)
            rec["lcq"].append(nan4)
            rec["rcp"].append(nan3)
            rec["rcq"].append(nan4)
            rec["lgc"].append(float(GRIPPER_UNKNOWN))
            rec["rgc"].append(float(GRIPPER_UNKNOWN))
            rec["lgp"].append(float("nan"))
            rec["rgp"].append(float("nan"))
            rec["jp"].append([float("nan")] * len(ARM_JOINTS) if args.joints else [])
            rec["jv"].append([float("nan")] * len(ARM_JOINTS) if args.joints else [])
            snap = pre_obj[k]
            for prim in track_prims:
                label = prim_label(prim)
                cur = snap.get(label)
                if cur is None:
                    obj_rec[label]["pos"].append(nan3)
                    obj_rec[label]["quat"].append(nan4)
                else:
                    obj_rec[label]["pos"].append(cur[0])
                    obj_rec[label]["quat"].append(cur[1])
    # 首次拿到 joint_states 时锁定关节顺序（按名字取，避免顺序依赖）
    # 关节列序固定为 ARM_JOINTS（不随 joint_states 到来说明变化）。
    # 缺失的关节在数组里是 NaN，保证每帧列数一致。
    joint_order: list[str] = list(ARM_JOINTS) if args.joints else []
    # 统计被识别为「无效 target」（全零 / 四元数退化为 0）并已回退为实测的帧数
    bad_target_frames = 0

    def _pose6(msg) -> tuple[list[float], list[float]]:
        p = msg.pose.position
        o = msg.pose.orientation
        return ([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])

    try:
        while True:
            # 睡到下一个采样时刻。若已经落后超过一个周期（比如 ROS 时钟跳变或
            # 处理超时），直接对齐到当前时刻，避免 next_t 越追越远导致 dt 抖动。
            now = pace.now()
            if next_t < now - period:
                next_t = now
            pace.sleep_until(next_t)
            next_t += period

            lp = latest["lp"].get()
            rp = latest["rp"].get()
            # 末端位姿可能比 /clock 晚几百毫秒（运控初始化）。
            # 此时不要跳过整帧 —— 否则物体沉降段也会丢。用 NaN 占位。

            # ROS 时间是分块推进的：时钟没动就不要落帧，否则会采到重复时间戳
            t_now = pace.now() - t_origin
            if rec["t"] and t_now <= rec["t"][-1]:
                continue
            js = latest["js"].get()
            lgp = rgp = float("nan")
            jpos: list[float] = []
            jvel: list[float] = []
            if js is not None:
                idx = {n: i for i, n in enumerate(js.name)}
                if LEFT_GRIP_JOINT in idx:
                    lgp = float(js.position[idx[LEFT_GRIP_JOINT]])
                if RIGHT_GRIP_JOINT in idx:
                    rgp = float(js.position[idx[RIGHT_GRIP_JOINT]])
                # 关节角/速度：列序固定为 ARM_JOINTS，缺失填 NaN。
                #
                # 不能「首次见到 joint_states 才锁定列序」——那样锁定前的帧是
                # 空数组 []、锁定后是 20 个值，np.asarray 会报
                # "inhomogeneous shape"。固定列序后每帧长度一致。
                if args.joints:
                    for n in ARM_JOINTS:
                        i_n = idx.get(n)
                        if i_n is None:
                            jpos.append(float("nan"))
                            jvel.append(float("nan"))
                        else:
                            jpos.append(float(js.position[i_n]))
                            jvel.append(
                                float(js.velocity[i_n]) if i_n < len(js.velocity) else 0.0
                            )
            elif args.joints:
                # 该帧还没有 /joint_states：整行 NaN，保持列数一致
                jpos = [float("nan")] * len(ARM_JOINTS)
                jvel = [float("nan")] * len(ARM_JOINTS)

            # 夹爪开合：优先用关节实测位置反推。
            # /left_gripper_controller/target_command 实际没有发布者（见
            # grip_from_position 注释），所以拿它的值只会得到 -1。
            lgc_cmd = latest["lgc"].get()
            rgc_cmd = latest["rgc"].get()
            lgc = float(lgc_cmd) if lgc_cmd is not None else grip_from_position(lgp)
            rgc = float(rgc_cmd) if rgc_cmd is not None else grip_from_position(rgp)
            # 若话题值本身是未知哨兵，也退回位置判定
            if lgc < 0:
                lgc = grip_from_position(lgp)
            if rgc < 0:
                rgc = grip_from_position(rgp)
            rec["lgc"].append(lgc)
            rec["rgc"].append(rgc)
            rec["lgp"].append(lgp)
            rec["rgp"].append(rgp)
            rec["jp"].append(jpos)
            rec["jv"].append(jvel)

            # 交互物体位姿：异步轮询，不阻塞。返回的是最近一次成功的值，
            # 查不到就沿用上一帧（避免 NaN 污染），首帧失败则记 NaN 占位。
            if poller is not None:
                obj_latest.update(poller.tick())
            for prim in track_prims:
                label = prim_label(prim)
                cur = obj_latest.get(label)
                if cur is None:
                    obj_rec[label]["pos"].append([float("nan")] * 3)
                    obj_rec[label]["quat"].append([float("nan")] * 4)
                else:
                    obj_rec[label]["pos"].append(cur[0])
                    obj_rec[label]["quat"].append(cur[1])

            rec["t"].append(t_now)
            nan3, nan4 = [float("nan")] * 3, [float("nan")] * 4
            if lp is not None:
                rec["lp"].append([lp.pose.position.x, lp.pose.position.y, lp.pose.position.z])
                rec["lq"].append([lp.pose.orientation.x, lp.pose.orientation.y,
                                  lp.pose.orientation.z, lp.pose.orientation.w])
            else:
                rec["lp"].append(nan3)
                rec["lq"].append(nan4)
            if rp is not None:
                rec["rp"].append([rp.pose.position.x, rp.pose.position.y, rp.pose.position.z])
                rec["rq"].append([rp.pose.orientation.x, rp.pose.orientation.y,
                                  rp.pose.orientation.z, rp.pose.orientation.w])
            else:
                rec["rp"].append(nan3)
                rec["rq"].append(nan4)

            # 指令目标（action）。
            #
            # 坑1：运控在「进入 HOLD / 臂 disable 重置」时会短暂把 target 缓存清成
            #      全零并发布出去（本次实测有 16 帧左右左右同时为 0）。这类帧不是
            #      有效动作，必须丢弃——否则训练会学到"跳回世界原点"。
            # 坑2：话题尚未发布时退回实测，保证两套数组等长。
            lct = latest["lct"].get()
            rct = latest["rct"].get()
            lcp_, lcq_ = _pose6(lct) if lct is not None else (rec["lp"][-1], rec["lq"][-1])
            rcp_, rcq_ = _pose6(rct) if rct is not None else (rec["rp"][-1], rec["rq"][-1])
            if not valid_target_pose(lcp_, lcq_):
                lcp_, lcq_ = rec["lp"][-1], rec["lq"][-1]
                bad_target_frames += 1
            if not valid_target_pose(rcp_, rcq_):
                rcp_, rcq_ = rec["rp"][-1], rec["rq"][-1]
                bad_target_frames += 1
            rec["lcp"].append(lcp_)
            rec["lcq"].append(lcq_)
            rec["rcp"].append(rcp_)
            rec["rcq"].append(rcq_)

            if args.duration > 0 and (pace.now() - t_origin) >= args.duration:
                break
            if args.max_frames > 0 and len(rec["t"]) >= args.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[info] 收到 Ctrl+C，结束采集")

    _shutdown(rclpy, executor, node, spin_thread)

    if not rec["t"]:
        print("[ERROR] 没有采到任何帧", file=sys.stderr)
        return 1

    traj = Traj(
        t=np.asarray(rec["t"], dtype=np.float64),
        left_pos=np.asarray(rec["lp"], dtype=np.float64),
        left_quat=np.asarray(rec["lq"], dtype=np.float64),
        right_pos=np.asarray(rec["rp"], dtype=np.float64),
        right_quat=np.asarray(rec["rq"], dtype=np.float64),
        left_grip=np.asarray(rec["lgc"], dtype=np.float64),
        right_grip=np.asarray(rec["rgc"], dtype=np.float64),
        left_grip_pos=np.asarray(rec["lgp"], dtype=np.float64),
        right_grip_pos=np.asarray(rec["rgp"], dtype=np.float64),
        frame_id=frame_id,
        left_cmd_pos=np.asarray(rec["lcp"], dtype=np.float64),
        left_cmd_quat=np.asarray(rec["lcq"], dtype=np.float64),
        right_cmd_pos=np.asarray(rec["rcp"], dtype=np.float64),
        right_cmd_quat=np.asarray(rec["rcq"], dtype=np.float64),
        joint_pos=(np.asarray(rec["jp"], dtype=np.float64) if joint_order else None),
        joint_vel=(np.asarray(rec["jv"], dtype=np.float64) if joint_order else None),
        joint_names=(list(joint_order) if joint_order else None),
        meta={
            "kind": "ee_trajectory",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hz": hz,
            "interrupted": interrupted,
            "robot": args.robot,
            "task": args.task,
            "note": args.note,
            "source_topics": {
                "left_pose": args.left_pose_topic,
                "right_pose": args.right_pose_topic,
                "left_target": TOPIC_LEFT_CUR_TGT,
                "right_target": TOPIC_RIGHT_CUR_TGT,
                "left_gripper": args.left_gripper_topic,
                "right_gripper": args.right_gripper_topic,
                "objects": "/get_entity_state" if track_prims else "",
            },
            "tracked_prims": {prim_label(p_): p_ for p_ in track_prims},
            # 哪些物体是刚体（回放时需设位姿）。静态物体只记录、不设位姿。
            "dynamic_objects": sorted(dynamic_labels),
        },
    )

    # 把物体位姿挂到 traj 上（Traj.save 会按 obj/<label>_{pos,quat} 写盘）
    if track_prims:
        traj.objects = {
            label: (np.asarray(d["pos"], dtype=np.float64),
                    np.asarray(d["quat"], dtype=np.float64))
            for label, d in obj_rec.items()
        }

    out = traj.save(args.out)
    print(f"\n[OK] 已保存 {len(traj)} 帧 → {out}")
    print(traj.summary())
    if poller is not None:
        msg = poller.failure_summary()
        if msg:
            print(f"[warn] {msg}")
    return 0


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    traj = Traj.load(args.file)
    print(f"文件: {Path(args.file).expanduser()}")
    print(traj.summary())
    if traj.meta:
        print("\nmeta:")
        for k, v in traj.meta.items():
            print(f"  {k}: {v}")

    # 逐段夹爪切换（方便对照编排块）
    for side, cmd in (("left", traj.left_grip), ("right", traj.right_grip)):
        idx = np.where(np.abs(np.diff(cmd)) > 0.5)[0]
        if idx.size:
            print(f"\n{side} 夹爪切换点:")
            for i in idx:
                print(f"  t={traj.t[i + 1]:.3f}s  -> {'开' if cmd[i + 1] > 0.5 else '关'}")

    # 位姿是否在某段完全不动（= 该臂被 disable，指令是「保持」）
    for side, cpos in (("left", traj.action("left")[0]), ("right", traj.action("right")[0])):
        if cpos.shape[0] < 10:
            continue
        frozen = _frozen_spans(cpos, traj.t, tol=1e-6, min_span=0.5)
        if frozen:
            print(f"\n{side} 指令静止段（该臂可能处于 DISABLE 保持）:")
            for a, b in frozen:
                print(f"  t={a:.2f}s → {b:.2f}s ({b - a:.2f}s 不动)")
            print("  回放时这几段会把该臂按住不动；若要跳过请用 slice 掐掉。")

    # 速度检查：过大的单帧跳变说明数据有问题（或采样丢了）
    for side, pos in (("left", traj.left_pos), ("right", traj.right_pos)):
        if pos.shape[0] < 2:
            continue
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dt = np.diff(traj.t)
        dt[dt <= 1e-6] = 1e-6
        v = d / dt
        i = int(np.argmax(v))
        print(f"\n{side} 最大瞬时速度 {v[i]:.3f} m/s @ t={traj.t[i + 1]:.3f}s"
              f"（单帧位移 {d[i] * 1000:.1f} mm）")
        if v[i] > 2.0:
            print("  [warn] 超过 2 m/s，可能是采样丢帧或位置跳变，回放时会触发运控急停/大速度。")

    # 关节角检查
    if traj.joint_pos is not None:
        jp = traj.joint_pos
        print(f"\n关节角: {jp.shape[1]} 个关节")
        names = traj.joint_names or [f"j{i}" for i in range(jp.shape[1])]
        tot = np.abs(np.diff(jp, axis=0)).sum(axis=0) if jp.shape[0] > 1 else np.zeros(jp.shape[1])
        for i in np.argsort(-tot)[:6]:
            rng = jp[:, i].max() - jp[:, i].min()
            unit = "m" if "LZ_" in names[i] else "°"
            scale = 1.0 if "LZ_" in names[i] else 180 / math.pi
            print(f"  {names[i]:30s} 幅度 {rng * scale:7.2f}{unit}  末端值 {jp[-1, i] * scale:7.2f}")
    else:
        print("\n[note] 本文件没有关节角数据（采集时 --no-joints 或没收到 /joint_states）")
    return 0


def cmd_slice(args: argparse.Namespace) -> int:
    """按帧或按时间截取子段，便于分段验证。"""
    traj = Traj.load(args.file)
    n = len(traj)
    t0 = float(traj.t[0])

    start = args.start
    end = args.end
    if args.start_sec is not None:
        start = int(np.searchsorted(traj.t - t0, args.start_sec))
    if args.end_sec is not None:
        end = int(np.searchsorted(traj.t - t0, args.end_sec))

    sub = traj.slice(start, end)
    if len(sub) < 2:
        print(f"[ERROR] 截取后只有 {len(sub)} 帧", file=sys.stderr)
        return 1

    out = args.out
    if not out:
        p = Path(args.file).expanduser()
        out = str(p.with_name(f"{p.stem}_s{start}_e{start + len(sub)}{p.suffix}"))

    sub.save(out)
    print(f"[OK] 已截取 帧[{start}:{start + len(sub)}] → {out}")
    print(sub.summary())
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """不采集、不回放，只报告回放前置条件。"""
    ros = _import_ros()
    rclpy = ros["rclpy"]
    PoseStamped = ros["PoseStamped"]

    rclpy.init()
    node = rclpy.create_node("ee_traj_check")

    latest: dict[str, _Latest] = {k: _Latest() for k in ("lp", "rp")}
    subs = [
        node.create_subscription(PoseStamped, args.left_pose_topic,
                                 lambda m: latest["lp"].set(m), 10),
        node.create_subscription(PoseStamped, args.right_pose_topic,
                                 lambda m: latest["rp"].set(m), 10),
    ]

    # WBC 状态：优先用消息类型；装不上就退回原始字段读取
    wbc_state: dict[str, Any] = {}

    def on_state(msg) -> None:
        wbc_state["msg"] = msg
        wbc_state["t"] = time.monotonic()

    has_state = False
    # current_state 用 transient_local（见 WbcCtrlComponent.cpp:1124），
    # 默认 VOLATILE 订阅不兼容，收不到任何消息。
    try:
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
    except Exception:  # noqa: BLE001
        state_qos = 10

    try:
        from arms_ros2_control_msgs.msg import WbcCurrentState  # type: ignore
        subs.append(node.create_subscription(WbcCurrentState, TOPIC_WBC_STATE, on_state, state_qos))
        has_state = True
    except Exception:  # noqa: BLE001
        pass

    if not has_state:
        # 回退：用 rosidl 通用类型按名字取（不依赖 arms_ros2_control_msgs 被 Python 找到）
        try:
            from rosidl_runtime_py.utilities import get_message
            T = get_message("arms_ros2_control_msgs/msg/WbcCurrentState")
            subs.append(node.create_subscription(T, TOPIC_WBC_STATE, on_state, state_qos))
            has_state = True
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] 无法解析 WbcCurrentState: {exc}")

    # Pacer 需要 /clock，因此必须在 spin 启动之后再构造
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.1)

    pace = Pacer(node, args.use_sim_time)

    print("[info] 探测 3s ...")
    time.sleep(3.0)

    ok = True
    print(f"\n时间源        : {'ROS(/clock) ' + f'{pace.now():.2f}' if pace.using_ros_time else '墙钟'}")
    for name, key, topic in (("左臂末端", "lp", args.left_pose_topic),
                             ("右臂末端", "rp", args.right_pose_topic)):
        msg = latest[key].get()
        cnt = latest[key].count
        if msg is None:
            print(f"{name} ({topic}) : [FAIL] 无消息")
            ok = False
        else:
            p = msg.pose.position
            print(f"{name} ({topic}) : [OK] {cnt} 条, frame='{msg.header.frame_id}', "
                  f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
            if msg.header.frame_id not in (DEFAULT_FRAME, ""):
                print(f"    [info] frame='{msg.header.frame_id}'（回放会原样回发同一 frame）")

    if not has_state:
        print(f"WBC 状态 ({TOPIC_WBC_STATE}) : [SKIP] 未安装 arms_ros2_control_msgs")
    elif "msg" not in wbc_state:
        print(f"WBC 状态      : [FAIL] 3s 内无 {TOPIC_WBC_STATE}")
        ok = False
    else:
        m = wbc_state["msg"]
        base = "LOCKED" if m.base_state == 0 else "UNLOCKED"
        arm_l = "ENABLED" if m.left_arm_state == 1 else "DISABLED"
        arm_r = "ENABLED" if m.right_arm_state == 1 else "DISABLED"
        bim = "COUPLED" if m.bimanual_state == 1 else "INDEPENDENT"
        print(f"WBC 状态      : base={base} left={arm_l} right={arm_r} bimanual={bim}")
        if m.left_arm_state != 1 or m.right_arm_state != 1:
            print("    [warn] 有臂处于 DISABLED：该臂的 /left_target|/right_target 会被丢弃。")
            print("           回放前请先发 LEFT_ARM_ENABLE / RIGHT_ARM_ENABLE，或确认编排已恢复。")
        if m.bimanual_state == 1:
            print("    [warn] BIMANUAL_COUPLED：/right_target 会被忽略（左臂 leader 优先）。")
            print("           回放脚本默认会先发 ARMS_INDEPENDENT。")

    print(f"\n结论: {'可以回放' if ok else '环境未就绪'}")
    _shutdown(rclpy, executor, node, spin_thread)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 回放
# ---------------------------------------------------------------------------

class _Tracker:
    """回放期间统计「指令 vs 实测」误差。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cur: dict[str, np.ndarray] = {}
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def set(self, side: str, pos: np.ndarray, quat: np.ndarray) -> None:
        with self._lock:
            self._cur[side] = np.concatenate([pos, quat])

    def snapshot(self, side: str):
        with self._lock:
            v = self._cur.get(side)
            return None if v is None else v.copy()

    def add(self, lcmd: np.ndarray, rcmd: np.ndarray, lmea, rmea) -> None:
        if lmea is None or rmea is None:
            return
        with self._lock:
            self.samples.append((lcmd, rcmd, lmea, rmea))

    def report(self) -> str:
        if not self.samples:
            return "未采到实测位姿，无法评估跟踪误差（检查 /left_current_pose 是否在发）。"
        arr = np.asarray(self.samples)
        lines = []
        for i, name in enumerate(("左手", "右手")):
            cmd = arr[:, i]
            mea = arr[:, i + 2]
            ep = np.linalg.norm(cmd[:, :3] - mea[:, :3], axis=1)
            eq = _quat_angle(cmd[:, 3:], mea[:, 3:])
            lines.append(
                f"{name}: 位置误差 mean {ep.mean() * 1000:6.1f} mm  "
                f"p95 {np.percentile(ep, 95) * 1000:6.1f} mm  "
                f"max {ep.max() * 1000:6.1f} mm | "
                f"姿态误差 mean {np.degrees(eq.mean()):5.2f}°  "
                f"max {np.degrees(eq.max()):5.2f}°"
            )
        lines.append("")
        lines.append("判定参考：p95 位置误差 < 20 mm 且 max < 60 mm，"
                     "说明运控能跟住这段末端轨迹，数据可用于训练。")
        return "\n".join(lines)


def _frozen_spans(pos: np.ndarray, t: np.ndarray,
                  tol: float = 1e-6, min_span: float = 0.5) -> list[tuple[float, float]]:
    """找出位姿完全静止的连续区间（秒）。"""
    if pos.shape[0] < 2:
        return []
    moving = np.linalg.norm(np.diff(pos, axis=0), axis=1) > tol
    spans: list[tuple[float, float]] = []
    start: int | None = 0 if not moving[0] else None
    for i, m in enumerate(moving):
        if not m and start is None:
            start = i
        elif m and start is not None:
            if t[i] - t[start] >= min_span:
                spans.append((float(t[start]), float(t[i])))
            start = None
    if start is not None and t[-1] - t[start] >= min_span:
        spans.append((float(t[start]), float(t[-1])))
    return spans


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """两组 xyzw 四元数之间的夹角（弧度）。"""
    a = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-9)
    b = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-9)
    d = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def _stamp_from_seconds(sec: float):
    """把秒数转成 builtin_interfaces/Time，避免依赖节点时钟。"""
    from builtin_interfaces.msg import Time
    t = Time()
    t.sec = int(math.floor(sec))
    t.nanosec = int(round((sec - t.sec) * 1e9))
    if t.nanosec >= 1_000_000_000:
        t.sec += 1
        t.nanosec -= 1_000_000_000
    return t


def _pose_msg(Pose, pos: np.ndarray, quat: np.ndarray):
    m = Pose()
    m.position.x = float(pos[0])
    m.position.y = float(pos[1])
    m.position.z = float(pos[2])
    m.orientation.x = float(quat[0])
    m.orientation.y = float(quat[1])
    m.orientation.z = float(quat[2])
    m.orientation.w = float(quat[3])
    return m


def cmd_replay(args: argparse.Namespace) -> int:
    traj = Traj.load(args.file)
    if args.start or args.end is not None:
        traj = traj.slice(args.start, args.end or None)
    if len(traj) < 2:
        print(f"[ERROR] 轨迹太短（{len(traj)} 帧）", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"回放 {args.file}")
    print("=" * 78)
    print(traj.summary())
    print()

    ros = _import_ros()
    rclpy = ros["rclpy"]
    Pose = ros["Pose"]
    PoseStamped = ros["PoseStamped"]
    Path = ros["Path"]
    Int32 = ros["Int32"]
    String = ros["String"]

    rclpy.init()
    node = rclpy.create_node("ee_traj_replay")

    pub_l = node.create_publisher(Pose, args.left_target_topic, 1)
    pub_r = node.create_publisher(Pose, args.right_target_topic, 1)
    pub_dual = node.create_publisher(Path, TOPIC_DUAL_TGT_STAMPED, 1)
    pub_lg = node.create_publisher(Int32, args.left_gripper_topic, 10)
    pub_rg = node.create_publisher(Int32, args.right_gripper_topic, 10)
    pub_mode = node.create_publisher(String, TOPIC_MODE_COMMAND, 10)

    tracker = _Tracker()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def make_cb(side: str):
        def cb(msg: PoseStamped) -> None:
            p = msg.pose.position
            o = msg.pose.orientation
            tracker.set(side, np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        return cb

    if args.measure:
        node.create_subscription(PoseStamped, args.left_pose_topic, make_cb("left"), 10)
        node.create_subscription(PoseStamped, args.right_pose_topic, make_cb("right"), 10)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.1)

    # Pacer 需要 /clock，必须在 spin 启动之后构造
    pace = Pacer(node, args.use_sim_time)
    print(f"[info] 时间源: {'ROS(/clock)' if pace.using_ros_time else '墙钟(monotonic)'}")

    def send_mode(cmd: str) -> None:
        m = String()
        m.data = cmd
        pub_mode.publish(m)
        print(f"  → mode_command: {cmd}")

    # 决定回放数据源：指令目标（复现意图）或实测（复现实际运动）
    if args.source == "auto":
        use_cmd = traj.has_cmd()
    else:
        use_cmd = args.source == "cmd"
    if use_cmd and not traj.has_cmd():
        print("[warn] 指定 --source cmd，但文件没有指令目标；退回实测位姿")
        use_cmd = False

    if use_cmd:
        l_pos, l_quat = traj.left_cmd_pos, traj.left_cmd_quat
        r_pos, r_quat = traj.right_cmd_pos, traj.right_cmd_quat
        print("[info] 回放数据源: 指令目标 (*_current_target)")
        gap = traj.cmd_gap()
        print(f"       指令 vs 原实测: {gap}")
        print("       注意：原实测里控制器没能完全跟住的地方，指令里是存在的；")
        print("       回放时若运控也跟不住，说明是真极限，不是数据问题。")
    else:
        l_pos, l_quat = traj.left_pos, traj.left_quat
        r_pos, r_quat = traj.right_pos, traj.right_quat
        print("[info] 回放数据源: 实测位姿 (*_current_pose)")

    # ---- 前置：模式设置 ----
    if not args.no_mode_setup:
        print("\n[1] 设置运控模式")
        for c in (["ARMS_INDEPENDENT"] if args.independent else []):
            send_mode(c)
        for c in ([args.base_lock_command] if args.lock_base else []):
            send_mode(c)
        for c in ([args.extra_mode] if args.extra_mode else []):
            send_mode(c)
        time.sleep(0.3)

    # ---- 平滑走到起点 ----
    if args.approach_duration > 0:
        print(f"\n[2] 用 /dual_target/stamped 平滑走到起点"
              f"（{args.approach_duration:.1f}s moveL，避免第一帧直接跳变）")
        path = Path()
        path.header.stamp = _stamp_from_seconds(pace.now())
        path.header.frame_id = traj.frame_id
        for pos, quat in ((l_pos[0], l_quat[0]), (r_pos[0], r_quat[0])):
            ps = PoseStamped()
            ps.header.stamp = path.header.stamp
            ps.header.frame_id = traj.frame_id
            ps.pose = _pose_msg(Pose, pos, quat)
            path.poses.append(ps)
        pub_dual.publish(path)
        # 起点夹爪状态先摆好
        _publish_gripper(pub_lg, pub_rg, Int32, traj.left_grip[0], traj.right_grip[0])
        pace.sleep(args.approach_duration + args.approach_settle)
    else:
        print("\n[2] 跳过平滑接近（--approach-duration 0），首帧将直接从当前位姿跳到起点")

    # ---- 流式回放 ----
    rate = max(0.05, float(args.rate_scale))
    n = len(traj)
    print(f"\n[3] 流式回放 {n} 帧，rate_scale={rate:.2f}"
          f"{'（含实测跟踪统计）' if args.measure else ''}")

    t0 = pace.now()
    src_t0 = float(traj.t[0])
    last_lg = last_rg = None
    done = 0
    try:
        for i in range(n):
            target = t0 + (float(traj.t[i]) - src_t0) / rate
            pace.sleep_until(target)

            lpos, lquat = l_pos[i], l_quat[i]
            rpos, rquat = r_pos[i], r_quat[i]
            pub_l.publish(_pose_msg(Pose, lpos, lquat))
            pub_r.publish(_pose_msg(Pose, rpos, rquat))

            lg, rg = float(traj.left_grip[i]), float(traj.right_grip[i])
            changed = (last_lg is None or abs(lg - last_lg) > EPS or
                       last_rg is None or abs(rg - last_rg) > EPS)
            if changed:
                _publish_gripper(pub_lg, pub_rg, Int32, lg, rg)
                last_lg, last_rg = lg, rg

            if args.measure:
                tracker.add(np.concatenate([lpos, lquat]), np.concatenate([rpos, rquat]),
                            tracker.snapshot("left"), tracker.snapshot("right"))

            done = i + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                el = (traj.t[i] - src_t0) / rate
                print(f"    {done}/{n}  轨迹时刻 {traj.t[i] - src_t0:6.2f}s  "
                      f"(实时 {el:6.2f}s)")
    except KeyboardInterrupt:
        print(f"\n[info] 中断于第 {done}/{n} 帧")
        done = n  # 仍然进入报告

    # ---- 收尾 ----
    print(f"\n[4] 保持末帧 {args.hold:.1f}s")
    pace.sleep(args.hold)

    if args.measure and done:
        print("\n[5] 跟踪误差（指令 vs 实测）")
        print(tracker.report())

    _shutdown(rclpy, executor, node, spin_thread)
    print("\n[OK] 回放结束")
    return 0


def _publish_gripper(pub_l, pub_r, Int32, lg: float, rg: float) -> None:
    for pub, val in ((pub_l, lg), (pub_r, rg)):
        if val < 0:
            continue  # 未采集到命令，不动
        m = Int32()
        m.data = GRIPPER_OPEN if val > 0.5 else GRIPPER_CLOSE
        pub.publish(m)
    print(f"    夹爪: left={_grip_word(lg)} right={_grip_word(rg)}")


def _grip_word(v: float) -> str:
    if v < 0:
        return "保持"
    return "开" if v > 0.5 else "关"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="末端轨迹采集/回放（EE-only，供 OCS2 全身控制验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--use-sim-time", choices=["auto", "ros", "wall"], default="auto",
                        help="计时源；auto=有 /clock 走 ROS 时间（默认）")
        sp.add_argument("--left-pose-topic", default=TOPIC_LEFT_CUR)
        sp.add_argument("--right-pose-topic", default=TOPIC_RIGHT_CUR)

    # record
    r = sub.add_parser("record", help="旁听并采集末端轨迹")
    add_common(r)
    r.add_argument("--out", "-o", required=True, help="输出 npz 路径")
    r.add_argument("--hz", type=float, default=50.0, help="采样频率（默认 50）")
    r.add_argument("--duration", type=float, default=0.0, help="采集时长秒；0=Ctrl+C 结束")
    r.add_argument("--max-frames", type=int, default=0, help="最大帧数；0=不限")
    r.add_argument("--wait-timeout", type=float, default=15.0)
    r.add_argument("--left-gripper-topic", default=TOPIC_LEFT_GRIP_CMD)
    r.add_argument("--right-gripper-topic", default=TOPIC_RIGHT_GRIP_CMD)
    r.add_argument("--robot", default="unitree_g1d")
    r.add_argument("--task", default="")
    r.add_argument("--note", default="")
    r.add_argument("--no-joints", dest="joints", action="store_false", default=True,
                   help="不采集关节角（默认采集，ACT 复刻需要）")
    r.add_argument(
        "--track-prims",
        default="",
        help=(
            "要跟踪位姿的交互物体 prim，逗号分隔。"
            "留空=用默认的试管与试管架；填 'none' 关闭。"
            "通过 /get_entity_state 查询，存入 obj/<label>_{pos,quat}。"
        ),
    )
    r.add_argument(
        "--from-sim-start",
        dest="from_sim_start",
        action="store_true",
        default=True,
        help=(
            "从仿真当前时刻立即开始采样（默认开）。配合「先启动本脚本、再按 Play」"
            "可以录到试管等物体的初始物理沉降。"
        ),
    )
    r.set_defaults(func=cmd_record)

    # inspect
    i = sub.add_parser("inspect", help="只读检查 npz（不需要 ROS）")
    i.add_argument("file")
    i.set_defaults(func=cmd_inspect)

    # slice
    sl = sub.add_parser("slice", help="按帧/时间截取子段（不需要 ROS）")
    sl.add_argument("file")
    sl.add_argument("--start", type=int, default=0, help="起始帧")
    sl.add_argument("--end", type=int, default=None, help="结束帧（不含）")
    sl.add_argument("--start-sec", type=float, default=None, help="起始秒（覆盖 --start）")
    sl.add_argument("--end-sec", type=float, default=None, help="结束秒（覆盖 --end）")
    sl.add_argument("--out", "-o", default="", help="输出路径；默认在原名后加 _s{a}_e{b}")
    sl.set_defaults(func=cmd_slice)

    # check
    c = sub.add_parser("check", help="回放前置条件检查")
    add_common(c)
    c.set_defaults(func=cmd_check)

    # replay
    rp = sub.add_parser("replay", help="把末端轨迹回放给运控")
    add_common(rp)
    rp.add_argument("file")
    rp.add_argument("--left-target-topic", default=TOPIC_LEFT_TGT)
    rp.add_argument("--right-target-topic", default=TOPIC_RIGHT_TGT)
    rp.add_argument("--left-gripper-topic", default=TOPIC_LEFT_GRIP_CMD)
    rp.add_argument("--right-gripper-topic", default=TOPIC_RIGHT_GRIP_CMD)
    rp.add_argument("--rate-scale", type=float, default=1.0,
                    help=">1 更快，<1 更慢（默认 1.0 = 原速）")
    rp.add_argument("--source", choices=["auto", "cmd", "meas"], default="auto",
                    help="回放哪套位姿：auto=有指令目标就用指令（推荐）；"
                         "cmd=只做意图复现；meas=复现原实测轨迹")
    rp.add_argument("--start", type=int, default=0, help="起始帧")
    rp.add_argument("--end", type=int, default=None, help="结束帧（不含）")
    rp.add_argument("--approach-duration", type=float, default=3.0,
                    help="先用 /dual_target/stamped 走到起点的时间；0=不做")
    rp.add_argument("--approach-settle", type=float, default=0.5)
    rp.add_argument("--hold", type=float, default=1.0, help="播放完保持末帧的秒数")
    rp.add_argument("--measure", dest="measure", action="store_true", default=True,
                    help="统计跟踪误差（默认开）")
    rp.add_argument("--no-measure", dest="measure", action="store_false")
    rp.add_argument("--progress-every", type=int, default=50, help="每 N 帧打印一次进度")
    rp.add_argument("--lock-base", action="store_true", default=True,
                    help="回放前发 BASE_LOCK（默认开：数据本身在 world 绝对系，底盘锁住最接近部署）")
    rp.add_argument("--no-lock-base", dest="lock_base", action="store_false")
    rp.add_argument("--base-lock-command", default="BASE_LOCK")
    rp.add_argument("--independent", action="store_true", default=True,
                    help="回放前发 ARMS_INDEPENDENT（默认开；耦合会丢 right_target）")
    rp.add_argument("--no-independent", dest="independent", action="store_false")
    rp.add_argument("--extra-mode", default="", help="额外下发的模式命令，如 RIGHT_ARM_ENABLE")
    rp.add_argument("--no-mode-setup", action="store_true")
    rp.set_defaults(func=cmd_replay)

    return p


def main(argv: list[str] | None = None) -> int:
    # 行缓冲：日志和 stderr 混在一起时保持顺序（管道下默认是块缓冲）
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
