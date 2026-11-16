#!/usr/bin/env python3
"""推理链路诊断 —— 从「点 Play」到「推理跑起来」全程记录，写到文件方便回看。

## 为什么不直接看终端
推理循环按 20Hz 打印 `[DEBUG]`，终端刷屏会把关键信息冲掉。这个脚本把
**每个阶段的检查结果**单独记到日志文件，并打上时间戳与结论（OK / FAIL）。

## 它检查什么（按时间顺序）

  A. 环境（Play 之前）
     - ROS2 环境、rmw、domain id
     - 关键话题是否存在、类型对不对
     - ocs2_wbc_controller 的 current_state 能否读到

  B. 机器人连接
     - ROS2Robot.connect() 是否成功
     - 相机话题是否有数据（拿到首帧的时间）

  C. 观测健全性（**最容易出错的地方**）
     - observation.state 的维度、**非零值个数**、取值范围
       （曾经因为关节名不匹配导致 32 维全零，策略输出退化成常数）
     - 三路图像是否拿到、shape、均值/方差（全黑/全白要能看出来）
     - 末端位姿是否有动态值

  D. mode 命令确认（BASE_LOCK / ARM_ENABLE）
     - 每条命令发出去后，轮询 current_state 直到确认或超时
     - 打印期望值与实际值，失败时明确说哪一项不匹配

  E. 首帧推理
     - 归一化前后的 state 范围
     - 模型输出（反归一化后）的范围
     - 与当前末端位姿的差（Delta）
     - 如果 Delta 很大或输出异常，直接标出来

  F. 少量步数的实际下发
     - 只发 N 帧（默认 30），每帧记录目标与实测
     - 看实测是否跟随目标（跟随 = 运控在工作；不跟随 = 运控没执行）

## 用法

```bash
cd ~/lerobot_ros2/examples/IsaacSim
source ~/lerobot_ros2/.venv-train/bin/activate

# 先把仿真 Play 起来（运控 + 相机话题正常），再运行：
python diag_inference.py \
    --dataset ~/lerobot_datasets/tube_ee \
    --train-config ~/act_runs/tube_ee/checkpoints/020000/pretrained_model \
    --frames 30 2>&1 | tee /tmp/diag_inference.log
```

日志文件默认写到 `/tmp/sdg_infer_diag/<时间戳>/`，含：
  - `report.txt`   人读的阶段报告
  - `phases.json`  机器可读的全量数据
  - `obs_sample.json` 第一帧观测的完整键值（排查键名/维度问题）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMMON = ROOT / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

_PHASES: list[dict[str, Any]] = []
_T0 = time.monotonic()


def phase(name: str) -> None:
    print(f"\n{'='*70}\n[{_rel():6.1f}s] {name}\n{'='*70}", flush=True)


def _rel() -> float:
    return time.monotonic() - _T0


def record(name: str, ok: bool | None, detail: Any, note: str = "") -> None:
    """记一条检查结果。ok=None 表示只是信息。"""
    mark = {True: "OK  ", False: "FAIL", None: "info"}[ok]
    _PHASES.append({"t": round(_rel(), 2), "name": name, "ok": ok,
                    "detail": detail, "note": note})
    body = json.dumps(detail, ensure_ascii=False, default=str)
    if len(body) > 300:
        body = body[:300] + "…"
    print(f"  [{mark}] {name}" + (f" :: {body}" if body != "{}" else "")
          + (f"   << {note}" if note else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="推理链路诊断（分阶段记录，避免刷屏）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--robot", default="unitree_g1d")
    ap.add_argument("--task", default="pick_tube_full")
    ap.add_argument("--frames", type=int, default=30, help="实际下发多少帧后停")
    ap.add_argument("--hz", type=float, default=10.0, help="诊断用低频，便于看跟随")
    ap.add_argument("--out", default="", help="输出目录")
    ap.add_argument("--n-action-steps", dest="n_action_steps", type=int, default=25,
                    help="覆盖 checkpoint 的 n_action_steps（默认 25）。"
                         "checkpoint 常是 1，配 chunk_size=100 会因运控稳态残差"
                         "导致目标每帧倒退。设 0 = 用原值。")
    args = ap.parse_args()

    out = Path(args.out or f"/tmp/sdg_infer_diag/{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"诊断输出目录: {out}")

    # ---------- A. 环境 ----------
    phase("A. 环境与话题")
    import os

    env = {k: os.environ.get(k, "") for k in
           ("ROS_DISTRO", "RMW_IMPLEMENTATION", "ROS_DOMAIN_ID",
            "ZENOH_SESSION_CONFIG_URI", "AMENT_PREFIX_PATH")}
    env["AMENT_PREFIX_PATH"] = env["AMENT_PREFIX_PATH"][:120]
    record("ros_env", None, env)

    try:
        import rclpy
        from rclpy.node import Node
        record("import_rclpy", True, {"version": getattr(rclpy, "__version__", "?")})
    except Exception as exc:  # noqa: BLE001
        record("import_rclpy", False, {"error": str(exc)})
        _dump(out, args)
        return 2

    rclpy.init()
    node = Node("sdg_infer_diag")

    # 等 topic 列表（Play 后运控才有话题）
    topics: list[tuple[str, list[str]]] = []
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        topics = [(n, t) for n, t in node.get_topic_names_and_types()]
        if topics:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    record("topic_count", bool(topics), {"n": len(topics)})

    want = ["/joint_states", "/left_target", "/right_target",
            "/dual_target/stamped", "/head_camera/rgb",
            "/left_hand_camera/rgb", "/right_hand_camera/rgb"]
    found = {n: (t or ["?"])[0] for n, t in topics}
    present = {w: found.get(w, None) for w in want}
    missing = [w for w, v in present.items() if v is None]
    record("expected_topics", not missing, present,
           note=f"缺 {missing}" if missing else "全部存在")

    # current_state
    cs_topic = "/ocs2_wbc_controller/current_state"
    record("current_state_topic", found.get(cs_topic) is not None,
           {"type": found.get(cs_topic)})

    # ---------- B. 连接机器人 ----------
    phase("B. 连接机器人 / 读观测")
    try:
        from robot_action_composer.discovery.registry_loader import (
            load_lerobot_profile, load_motion_entries,
            resolve_robot_profile_dir,
        )
        from online_infer import core as C

        ws = ROOT
        reg = load_motion_entries(ws)
        entry = reg[args.robot]
        motion_cfg = entry["motion_cfg"]
        lerobot_cfg = load_lerobot_profile(
            resolve_robot_profile_dir(ws, entry), workspace_dir=ws)
        C.MOTION_CFG = motion_cfg
        C.ROBOT_CFG = motion_cfg
        C.LEROBOT_CFG = lerobot_cfg
        record("load_config", True, {"robot": args.robot,
                                     "cameras": list(getattr(lerobot_cfg, "cameras", {}) or {})})
    except Exception as exc:  # noqa: BLE001
        record("load_config", False, {"error": repr(exc)})
        _dump(out, args)
        return 2

    try:
        robot = C.build_robot()
        robot.connect()
        record("robot_connect", True, {"connected": robot.is_connected})
    except Exception as exc:  # noqa: BLE001
        record("robot_connect", False, {"error": repr(exc)})
        _dump(out, args)
        return 2

    # 读一帧观测（等待画面）
    obs: dict[str, Any] = {}
    for attempt in range(40):            # 最多等 8s（40 × 0.2）
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            obs = robot.get_observation() or {}
        except Exception as exc:  # noqa: BLE001
            record("get_observation", False, {"error": repr(exc)})
            obs = {}
        if obs:
            break
        time.sleep(0.2)

    scalars = {k: v for k, v in obs.items() if isinstance(v, (int, float))}
    arrays = {k: (getattr(v, "shape", None), str(getattr(v, "dtype", "")))
              for k, v in obs.items() if hasattr(v, "shape")}
    record("obs_keys", bool(obs), {"scalar_n": len(scalars), "array_n": len(arrays),
                                   "sample_array": dict(list(arrays.items())[:6])})
    (out / "obs_sample.json").write_text(
        json.dumps({"scalars": scalars, "arrays": arrays},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 关节非零检查（曾因名字不匹配全零）
    joint_pos = {k: v for k, v in scalars.items() if k.endswith(".pos")}
    nonzero = [k for k, v in joint_pos.items() if abs(float(v)) > 1e-9]
    record("joint_pos_nonzero", len(nonzero) > 0,
           {"total": len(joint_pos), "nonzero": len(nonzero),
            "zero_ones": [k for k in joint_pos if k not in nonzero][:8]},
           note="若 nonzero=0 说明关节名没对上，state 全零会让策略退化")

    # 图像检查
    imgs = {k: v for k, v in obs.items() if hasattr(v, "shape") and getattr(v, "ndim", 0) == 3}
    img_stat = {}
    for k, v in imgs.items():
        import numpy as np
        a = np.asarray(v, dtype=np.float32)
        img_stat[k] = {"shape": list(a.shape), "mean": round(float(a.mean()), 2),
                       "std": round(float(a.std()), 2)}
    record("images", len(imgs) > 0, img_stat,
           note="std=0 说明是纯色（相机没出图）" if any(
               s["std"] < 0.5 for s in img_stat.values()) else "")

    ee = {k: v for k, v in scalars.items() if k.startswith(("left_ee", "right_ee"))}
    record("ee_pose", bool(ee), {k: round(float(v), 5) for k, v in list(ee.items())[:14]})

    # ---------- C. mode 命令确认 ----------
    phase("C. mode 命令确认（BASE_LOCK / ARM_ENABLE）")
    iface = robot.ros2_interface

    # ---- 末端目标走的是哪个话题（**排查"发了但没人听"的第一手信息**）----
    topic_info = {}
    for side, h in (("left", getattr(iface, "left_arm_handler", None)),
                    ("right", getattr(iface, "right_arm_handler", None))):
        if h is None:
            topic_info[side] = {"handler": None}
            continue
        topic_info[side] = {
            "target_topic(纯Pose)": getattr(h, "target_topic", None),
            "has_send_target": hasattr(h, "send_target"),
            "has_send_target_stamped": hasattr(h, "send_target_stamped"),
            "pub_ready": getattr(h, "target_pub", None) is not None,
            "stamped_pub_ready": getattr(h, "target_stamped_pub", None) is not None,
        }
    record("ee_target_topics", None, topic_info,
           note="send_target 发纯 Pose 到 <topic>；send_target_stamped 发 PoseStamped 到 <topic>/stamped")

    # ---- 运控是否**收到**了目标（/left_current_target 是运控回读的内部目标）----
    #
    # 这是区分「消息没发出去」和「运控没接受」的关键：
    #   我们发 /left_target（指令）→ 运控处理后会在 /left_current_target
    #   回读它当前跟踪的目标。两者都动 = 运控在工作；只有前者动 = 运控忽略了。
    ct_topic = "/left_current_target"
    record("current_target_topic_exists", ct_topic in dict(node.get_topic_names_and_types()),
           {"topic": ct_topic})

    record("fsm_command", None, {"before": _read(cs_topic, node, rclpy)})

    def send_and_wait(cmds: list[str], timeout: float = 5.0) -> bool:
        expect = {c: iface.MODE_COMMAND_TO_WBC_EXPECT.get(c) for c in cmds}
        record(f"send {cmds}", None, {c: e for c, e in expect.items()})
        for c in cmds:
            try:
                iface.send_mode_command(c)
            except Exception as exc:  # noqa: BLE001
                record(f"send {c}", False, {"error": repr(exc)})
                return False
        t0 = time.monotonic()
        matched = False
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                matched = bool(iface.mode_commands_match_wbc_state(cmds))
            except Exception as exc:  # noqa: BLE001
                record("mode_match", False, {"error": repr(exc)})
                return False
            if matched:
                break
            time.sleep(0.05)
        state = _read(cs_topic, node, rclpy)
        record(f"confirm {cmds}", matched,
               {"elapsed_s": round(time.monotonic() - t0, 2), "current_state": state},
               note="" if matched else "期望值见上，actual 见 current_state 各字段")
        return matched

    ok_lock = send_and_wait(["BASE_LOCK"], 5.0)
    ok_en = send_and_wait(["LEFT_ARM_ENABLE", "RIGHT_ARM_ENABLE"], 5.0)
    record("mode_summary", ok_lock and ok_en,
           {"base_lock": ok_lock, "arm_enable": ok_en},
           note="任一 False → 机器人不会跟随末端目标")

    # ---------- D. 加载策略 + 首帧推理 ----------
    phase("D. 加载策略与首帧推理")
    try:
        C.PICK_PLACE_FLOW_OVERRIDES = {"object_prim_path": "", "no_reset": True}
        ns = argparse.Namespace(
            dataset=Path(args.dataset), train_config=Path(args.train_config),
            checkpoint=None, device="cuda", tolerance_s=None, hz=args.hz,
            debug=False, relative=False, fixed_gripper=None, auto_close_z=None,
            no_state=False, drop_state_names="",
            n_action_steps=args.n_action_steps,
        )
        _tc, meta, policy, img_feats, gripper_idx, pre, post, right_gripper_idx = C.init_inference(ns)
        record("policy_loaded", True, {
            "image_feats": img_feats, "gripper_idx": gripper_idx,
            "right_gripper_idx": right_gripper_idx,
            "has_pre": pre is not None, "has_post": post is not None,
            "state_names": (meta.features.get("observation.state", {}) or {}).get("names", [])[:6],
            "action_names": (meta.features.get("action", {}) or {}).get("names", []),
        })
    except Exception as exc:  # noqa: BLE001
        import traceback
        record("policy_loaded", False, {"error": repr(exc)})
        traceback.print_exc()
        _dump(out, args)
        return 2

    # 首帧推理（拆开看归一化前后）
    import numpy as np

    sv_raw = C.build_state_vector(meta, obs)
    record("state_raw", None, {"dim": int(sv_raw.size),
                               "min": round(float(sv_raw.min()), 5),
                               "max": round(float(sv_raw.max()), 5),
                               "nonzero": int((np.abs(sv_raw) > 1e-9).sum())})

    rec = C.infer_action_vec(
        policy=policy, meta=meta, policy_image_feats=img_feats,
        raw_obs=obs, device="cuda", hz=args.hz,
        sleep_fn=time.sleep, preprocessor=pre, postprocessor=post,
    )
    if rec is None:
        record("first_infer", False, {"error": "infer_action_vec 返回 None（多半缺图像）"})
        _dump(out, args)
        return 2
    rec = np.asarray(rec).reshape(-1)
    record("first_infer", True, {"dim": int(rec.size),
                                 "min": round(float(rec.min()), 4),
                                 "max": round(float(rec.max()), 4)})

    # 与当前末端比
    action_names = (meta.features.get("action", {}) or {}).get("names", [])
    cmd = C.build_cmd_from_action_vec(meta, rec, obs, ns, gripper_idx, right_gripper_idx)
    diff = {}
    for axis in "xyz":
        k = f"left_ee.pos.{axis}"
        cur = obs.get(k)
        tgt = cmd.get(k)
        if cur is not None and tgt is not None:
            diff[axis] = round(float(tgt) - float(cur), 5)
    dist = float(np.linalg.norm(list(diff.values()))) if diff else None
    record("first_cmd_vs_current", None,
           {"current": {a: round(float(obs.get(f"left_ee.pos.{a}", 0)), 4) for a in "xyz"},
            "target": {a: round(float(cmd.get(f"left_ee.pos.{a}", 0)), 4) for a in "xyz"},
            "delta": diff, "dist_m": round(dist, 4) if dist else None},
           note="dist > 0.5m 基本可以断定坐标系/归一化有问题" if (dist or 0) > 0.5 else "")

    # ---------- E. 实际下发 N 帧，看跟随 ----------
    phase(f"E. 下发 {args.frames} 帧（{args.hz}Hz）看实测是否跟随目标")

    # 订阅运控回读的内部目标（判断它到底有没有接受我们的指令）
    _cur_tgt_box: dict[str, list] = {"v": []}
    try:
        from rosidl_runtime_py.utilities import get_message as _gm2

        _ts = dict(node.get_topic_names_and_types())
        if "/left_current_target" in _ts:
            def _ct_cb(m):
                # /left_current_target 可能是 Pose 或 PoseStamped，两种都兼容
                p = getattr(m, "pose", m)
                _cur_tgt_box["v"].append(
                    (float(p.position.x), float(p.position.y), float(p.position.z)))
            _ct_sub = node.create_subscription(
                _gm2(_ts["/left_current_target"][0]), "/left_current_target", _ct_cb, 10)
        else:
            _ct_sub = None
    except Exception:  # noqa: BLE001
        _ct_sub = None

    track = []
    for i in range(args.frames):
        rclpy.spin_once(node, timeout_sec=0.01)
        try:
            o = robot.get_observation() or {}
        except Exception:  # noqa: BLE001
            o = {}
        a = C.infer_action_vec(
            policy=policy, meta=meta, policy_image_feats=img_feats,
            raw_obs=o, device="cuda", hz=args.hz,
            sleep_fn=time.sleep, preprocessor=pre, postprocessor=post,
        )
        if a is None:
            continue
        cm = C.build_cmd_from_action_vec(meta, a, o, ns, gripper_idx)

        # 第一次下发前，先订阅 /left_target 和 /left_target/stamped，
        # 确认消息真的发出去了（"发了但没人听" vs "根本没发" 是两种完全
        # 不同的故障，必须分清）。
        if i == 0:
            try:
                from rosidl_runtime_py.utilities import get_message as _gm

                _types = dict(node.get_topic_names_and_types())
                _sniff: dict[str, int] = {}

                def _mk_cb(tag):
                    def cb(_msg):
                        _sniff[tag] = _sniff.get(tag, 0) + 1
                    return cb

                _subs = []
                for _t in ("/left_target", "/left_target/stamped",
                           "/dual_target/stamped"):
                    if _t in _types:
                        _subs.append(node.create_subscription(
                            _gm(_types[_t][0]), _t, _mk_cb(_t), 10))
                robot.send_action(cm)
                _t0 = time.monotonic()
                while time.monotonic() - _t0 < 1.0:
                    rclpy.spin_once(node, timeout_sec=0.1)
                for _s in _subs:
                    node.destroy_subscription(_s)
                record("publish_sniff", bool(_sniff), {"counts": _sniff},
                       note="哪个话题真有消息，就说明 send_action 走的是那条路；"
                            "若全为 0 说明根本没发布")
            except Exception as exc:  # noqa: BLE001
                record("publish_sniff", False, {"error": repr(exc)})

        robot.send_action(cm)
        cur = [float(o.get(f"left_ee.pos.{ax}", 0.0)) for ax in "xyz"]
        tgt = [float(cm.get(f"left_ee.pos.{ax}", 0.0)) for ax in "xyz"]
        track.append({"i": i, "cur": [round(v, 4) for v in cur],
                      "tgt": [round(v, 4) for v in tgt]})
        if i % 5 == 0 or i == args.frames - 1:
            d = float(np.linalg.norm(np.array(tgt) - np.array(cur)))
            print(f"  frame {i:2d}  cur={[round(v,3) for v in cur]}  "
                  f"tgt={[round(v,3) for v in tgt]}  |Δ|={d:.4f}m", flush=True)
        time.sleep(1.0 / max(args.hz, 1.0))

    if track:
        moved = float(np.linalg.norm(
            np.array(track[-1]["cur"]) - np.array(track[0]["cur"])))
        # ---- 方向检查（比"动没动"更关键）----
        #
        # 曾经误判过：机器人没跟随目标，只是重力下垂 8.8mm，而旧判据
        # 只看 moved > 2mm 就报 OK。实际位移向量与目标位移向量**方向相反**。
        #
        # 正确的判据：实测位移向量 (cur_last - cur_first) 与目标位移向量
        # (tgt_last - tgt_first) 的点积。点积 ≤ 0 = 方向相反 = 没跟随。
        d_cur = np.array(track[-1]["cur"]) - np.array(track[0]["cur"])
        d_tgt = np.array(track[-1]["tgt"]) - np.array(track[0]["tgt"])
        n_tgt = float(np.linalg.norm(d_tgt))
        n_cur = float(np.linalg.norm(d_cur))
        dot = float(np.dot(d_cur, d_tgt))
        if n_tgt < 1e-4:
            verdict = None                     # 目标没动，无法判方向
            note = "目标几乎没动（策略输出接近静止），无法判断跟随方向"
        elif n_cur < 1e-4:
            verdict = False
            note = "末端完全没动 —— 运控没执行目标（话题/mode 问题）"
        elif dot > 0:
            verdict = True
            cos = dot / max(n_cur * n_tgt, 1e-12)
            note = f"朝目标方向移动（cos={cos:.2f}）"
        else:
            verdict = False
            note = (f"**方向相反**（cos={dot/max(n_cur*n_tgt,1e-12):.2f}）"
                    f" —— 实测位移 {np.round(d_cur,4)} vs 目标位移 {np.round(d_tgt,4)}；"
                    f"末端只是受力下垂，运控没跟随")

        record("tracking_direction", verdict,
               {"first_cur": track[0]["cur"], "last_cur": track[-1]["cur"],
                "first_tgt": track[0]["tgt"], "last_tgt": track[-1]["tgt"],
                "moved_cur_m": round(n_cur, 5), "moved_tgt_m": round(n_tgt, 5),
                "dot": round(dot, 8)},
               note=note)

    # 运控内部目标是否跟着我们发的指令走
    if _ct_sub is not None:
        try:
            node.destroy_subscription(_ct_sub)
        except Exception:  # noqa: BLE001
            pass
    v = _cur_tgt_box["v"]
    if v:
        a, b = np.array(v[0]), np.array(v[-1])
        record("controller_current_target", float(np.linalg.norm(b - a)) > 1e-4,
               {"samples": len(v), "first": list(np.round(a, 4)),
                "last": list(np.round(b, 4)),
                "moved_m": round(float(np.linalg.norm(b - a)), 5)},
               note="运控内部目标动了 = 它接受了我们的指令；没动 = 忽略/未处理")
    elif _ct_sub is not None:
        record("controller_current_target", False, {"samples": 0},
               note="/left_current_target 存在但没收到消息 —— 运控没在跟踪末端目标")

    phase("诊断完成")
    _dump(out, args)
    print(f"\n日志: {out}/report.txt")
    return 0


def _read(topic: str, node, rclpy) -> dict:
    """读一次某个话题的最新消息（转成简单 dict）。"""
    box: dict[str, Any] = {}

    def cb(msg):
        box["msg"] = msg

    try:
        from rosidl_runtime_py.utilities import get_message
        types = dict(node.get_topic_names_and_types()).get(topic)
        if not types:
            return {"error": "topic 不存在"}
        sub = node.create_subscription(get_message(types[0]), topic, cb, 10)
        t0 = time.monotonic()
        while "msg" not in box and time.monotonic() - t0 < 1.5:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_subscription(sub)
        if "msg" not in box:
            return {"error": "1.5s 内没收到消息"}
        m = box["msg"]
        out = {}
        for f in getattr(m, "__slots__", []):
            v = getattr(m, f, None)
            if isinstance(v, (int, float, str, bool)):
                out[f] = v
            elif hasattr(v, "__slots__"):
                out[f] = {g: getattr(v, g) for g in v.__slots__
                          if isinstance(getattr(v, g, None), (int, float, str, bool))}
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _dump(out: Path, args) -> None:
    (out / "phases.json").write_text(
        json.dumps({"args": vars(args), "phases": _PHASES},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [f"推理链路诊断报告  ({time.strftime('%Y-%m-%d %H:%M:%S')})", ""]
    for p in _PHASES:
        mark = {True: "OK  ", False: "FAIL", None: "info"}[p["ok"]]
        lines.append(f"[{p['t']:7.2f}s] [{mark}] {p['name']}")
        lines.append(f"          {json.dumps(p['detail'], ensure_ascii=False, default=str)}")
        if p["note"]:
            lines.append(f"          << {p['note']}")
    lines.append("")
    fails = [p for p in _PHASES if p["ok"] is False]
    lines.append(f"失败项: {len(fails)}")
    for p in fails:
        lines.append(f"  - {p['name']}: {p['note'] or p['detail']}")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
