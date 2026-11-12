#!/usr/bin/env python3
"""对照实验：DISABLE 后等多久再 ENABLE 才能成功。

假设：DISABLE 需要等新一代 MPC 策略（WAITING_FOR_DISABLE_POLICY -> DISABLED_HOLD），
在此之前发 ENABLE 会被 "Mode rejected (arm hold transition in progress)" 整条丢弃。
"""
import argparse
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from rosidl_runtime_py.utilities import get_message


class Probe(Node):
    def __init__(self):
        super().__init__("delay_probe")
        self.state = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
        T = get_message("arms_ros2_control_msgs/msg/WbcCurrentState")
        self.create_subscription(T, "/ocs2_wbc_controller/current_state", self._cb, qos)
        self.pub = self.create_publisher(String, "/mode_command", 10)

    def _cb(self, m):
        self.state = m

    def send(self, c):
        msg = String()
        msg.data = c
        self.pub.publish(msg)

    def left(self):
        return None if self.state is None else self.state.left_arm_state

    def wait_left(self, want, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.left() == want:
                return time.monotonic() - t0
            time.sleep(0.02)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", default="0.3,0.5,1.0,2.0")
    ap.add_argument("--enable-timeout", type=float, default=12.0)
    args = ap.parse_args()

    rclpy.init()
    n = Probe()
    threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
    time.sleep(1.5)

    print(f"初始 left_arm_state={n.left()}")
    print(f"{'delay':>7} {'DISABLE':>10} {'ENABLE':>10} 备注")
    print("-" * 60)

    for d in [float(x) for x in args.delays.split(",")]:
        # 确保起始是 ENABLED
        if n.left() == 0:
            n.send("LEFT_ARM_ENABLE")
            if n.wait_left(1, args.enable_timeout) is None:
                print(f"{d:>7.1f} {'-':>10} {'残留DIS':>10} 无法复位，跳过")
                continue
            time.sleep(0.5)

        # DISABLE
        n.send("LEFT_ARM_DISABLE")
        t_dis = n.wait_left(0, 6.0)
        if t_dis is None:
            print(f"{d:>7.1f} {'FAIL':>10} {'-':>10} DISABLE 未确认")
            continue

        # 等待指定间隔后 ENABLE
        time.sleep(d)
        n.send("LEFT_ARM_ENABLE")
        t_en = n.wait_left(1, args.enable_timeout)

        if t_en is None:
            print(f"{d:>7.1f} {t_dis:>9.2f}s {'TIMEOUT':>10} ← 被拒/卡住")
        else:
            print(f"{d:>7.1f} {t_dis:>9.2f}s {t_en:>9.2f}s  OK")
        time.sleep(0.8)

    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
