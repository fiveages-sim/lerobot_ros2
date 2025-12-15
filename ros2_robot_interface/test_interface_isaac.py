"""
ROS2机器人接口测试脚本 - Isaac Sim联合仿真版本
用于验证夹爪关闭判断逻辑
"""

import time
import sys
from geometry_msgs.msg import Pose

from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig, ControlType


def main():
    """测试ROS2机器人接口（Isaac Sim联合仿真）"""
    
    # ========================================================================
    # 第一部分：初始化和连接
    # ========================================================================
    print("\n" + "=" * 70)
    print(" " * 20 + "ROS2 Robot Interface Test - Isaac Sim")
    print("=" * 70 + "\n")
    
    # 创建配置对象（模式将通过ROS 2 topic自动检测）
    print("[1] 创建配置...")
    print("    → 模式将通过ROS 2 topic自动检测")
    print("    → 检查是否存在 /right_target 或 /right_current_pose\n")
    config = ROS2RobotInterfaceConfig()
    
    # 创建并连接接口实例
    print("[2] 创建ROS2RobotInterface实例...")
    interface = ROS2RobotInterface(config)
    
    print("[3] 连接到ROS 2...")
    try:
        interface.connect()
        print("    ✓ 接口连接成功!\n")
    except Exception as e:
        print(f"    ✗ 连接失败: {e}\n")
        return 1
    
    # 等待数据到达（给ROS 2一些时间进行topic发现和数据传输）
    print("[4] 等待数据到达（2秒）...")
    time.sleep(2.0)
    print("    ✓ 数据收集已开始\n")
    
    # 切换到HOME状态（FSM循环的起始状态）
    print("-" * 70)
    print("[5] 切换到HOME状态（起始状态）")
    print("-" * 70)
    try:
        interface.send_fsm_command(1)  # 1 = HOME状态
        print("  ✓ FSM命令已发送: 切换到HOME状态")
        time.sleep(1.0)  # 等待状态转换完成
        print("  ✓ 状态转换完成\n")
    except Exception as e:
        print(f"  ⚠ 切换到HOME状态失败: {e}\n")
    
    # ========================================================================
    # 第二部分：测试数据获取功能
    # ========================================================================
    
    # 测试：获取关节状态（原始数据，所有关节）
    print("-" * 70)
    print("[6] 测试 get_joint_state() - 获取原始关节状态")
    print("-" * 70)
    joint_state = interface.get_joint_state()
    if joint_state:
        print(f"  ✓ 关节状态已接收")
        print(f"    总关节数: {len(joint_state['names'])}")
        print(f"    关节名称: {', '.join(joint_state['names'][:5])}{'...' if len(joint_state['names']) > 5 else ''}")
        print(f"    位置:   {[f'{p:.3f}' for p in joint_state['positions'][:5]]}{'...' if len(joint_state['positions']) > 5 else ''}")
        print(f"    速度:  {[f'{v:.3f}' for v in joint_state['velocities'][:5]]}{'...' if len(joint_state['velocities']) > 5 else ''}")
    else:
        print("  ⚠ 尚未接收到关节状态")
    print()
    
    # 测试：获取分类后的关节状态（按身体部位分类）
    print("-" * 70)
    print("[6.5] 测试 get_joint_state(categorized=True) - 获取分类关节状态")
    print("-" * 70)
    categorized_joint_state = interface.get_joint_state(categorized=True)
    if categorized_joint_state:
        is_dual_arm = interface.config.right_end_effector_pose_topic is not None
        
        if is_dual_arm:
            # Dual-arm mode
            if categorized_joint_state.get('left_arm', {}).get('names'):
                left_arm = categorized_joint_state['left_arm']
                print(f"  ✓ Left arm: {len(left_arm['names'])} joints")
                print(f"    Names: {', '.join(left_arm['names'][:5])}{'...' if len(left_arm['names']) > 5 else ''}")
                print(f"    Positions: {[f'{p:.3f}' for p in left_arm['positions'][:5]]}{'...' if len(left_arm['positions']) > 5 else ''}")
            
            if categorized_joint_state.get('right_arm', {}).get('names'):
                right_arm = categorized_joint_state['right_arm']
                print(f"  ✓ Right arm: {len(right_arm['names'])} joints")
                print(f"    Names: {', '.join(right_arm['names'][:5])}{'...' if len(right_arm['names']) > 5 else ''}")
                print(f"    Positions: {[f'{p:.3f}' for p in right_arm['positions'][:5]]}{'...' if len(right_arm['positions']) > 5 else ''}")
        else:
            # Single-arm mode
            if categorized_joint_state.get('arm', {}).get('names'):
                arm = categorized_joint_state['arm']
                print(f"  ✓ Arm: {len(arm['names'])} joints")
                print(f"    Names: {', '.join(arm['names'][:5])}{'...' if len(arm['names']) > 5 else ''}")
                print(f"    Positions: {[f'{p:.3f}' for p in arm['positions'][:5]]}{'...' if len(arm['positions']) > 5 else ''}")
        
        if categorized_joint_state.get('head', {}).get('names'):
            head = categorized_joint_state['head']
            print(f"  ✓ Head: {len(head['names'])} joints")
            print(f"    Names: {', '.join(head['names'])}")
            print(f"    Positions: {[f'{p:.3f}' for p in head['positions']]}")
        
        if categorized_joint_state.get('body', {}).get('names'):
            body = categorized_joint_state['body']
            print(f"  ✓ Body: {len(body['names'])} joints")
            print(f"    Names: {', '.join(body['names'])}")
            print(f"    Positions: {[f'{p:.3f}' for p in body['positions']]}")
        
        if categorized_joint_state.get('other', {}).get('names'):
            other = categorized_joint_state['other']
            print(f"  ⚠ Other: {len(other['names'])} joints")
            print(f"    Names: {', '.join(other['names'])}")
    else:
        print("  ⚠ No categorized joint state available")
    print()
    
    # 测试：获取末端执行器位姿（位置和方向）
    print("-" * 70)
    print("[7] 测试 get_end_effector_pose() - 获取末端执行器位姿")
    print("-" * 70)
    
    # 左臂/主臂位姿
    left_pose = interface.get_end_effector_pose()
    if left_pose:
        print(f"  ✓ 末端执行器位姿已接收")
        print(f"    位置:    ({left_pose.position.x:7.3f}, {left_pose.position.y:7.3f}, {left_pose.position.z:7.3f})")
        print(f"    方向:  ({left_pose.orientation.x:6.3f}, {left_pose.orientation.y:6.3f}, "
              f"{left_pose.orientation.z:6.3f}, {left_pose.orientation.w:6.3f})")
    else:
        print("  ⚠ 尚未接收到末端执行器位姿")
    
    # 右臂位姿（仅双臂模式）
    right_pose = interface.get_right_end_effector_pose()
    if right_pose:
        print(f"\n  ✓ 右臂位姿已接收")
        print(f"    位置:    ({right_pose.position.x:7.3f}, {right_pose.position.y:7.3f}, {right_pose.position.z:7.3f})")
        print(f"    方向:  ({right_pose.orientation.x:6.3f}, {right_pose.orientation.y:6.3f}, "
              f"{right_pose.orientation.z:6.3f}, {right_pose.orientation.w:6.3f})")
    elif interface.config.right_end_effector_pose_topic:
        print("\n  ⚠ 尚未接收到右臂位姿（已检测到双臂模式）")
    print()
    
    # ========================================================================
    # 第三部分：夹爪循环开关控制（用于Isaac Sim联合仿真验证）
    # ========================================================================
    print("=" * 70)
    print("[8] 夹爪循环开关控制")
    print("=" * 70)
    is_dual_arm = interface.config.right_end_effector_pose_topic is not None
    if is_dual_arm:
        print("  模式: 双臂")
        gripper_part = 'left_gripper'
    else:
        print("  模式: 单臂")
        gripper_part = 'gripper'
    print("  → 夹爪循环开关：关闭(0) → 打开(1) → 关闭(0) → ...")
    print("  → 关闭时检测：如果位置稳定且不在0，说明抓到东西，停止打开")
    print("  → 按Ctrl+C停止并断开连接")
    print("-" * 70 + "\n")
    
    # 初始夹爪状态检测（进入循环前先确认当前夹爪位置）
    initial_position = None
    categorized_state = interface.get_joint_state(categorized=True)
    if categorized_state:
        if is_dual_arm and 'left_gripper' in categorized_state:
            data = categorized_state['left_gripper']
            if data.get('positions'):
                initial_position = data['positions'][0]
        elif not is_dual_arm and 'gripper' in categorized_state:
            data = categorized_state['gripper']
            if data.get('positions'):
                initial_position = data['positions'][0]
    
    # 夹爪控制变量
    if initial_position is not None:
        # 位置<0.1认为闭合，>=0.1认为张开
        gripper_is_open = initial_position >= 0.1
        print(f"  → 初始夹爪位置: {initial_position:.4f}，状态判定为: {'张开' if gripper_is_open else '闭合'}")
    else:
        gripper_is_open = False
        print("  → 未获取到夹爪初始位置，默认按闭合处理")
    
    gripper_target_open = not gripper_is_open  # 下一个目标状态
    gripper_command_sent = False  # 是否已发送命令，等待到达
    object_detected = False  # 是否检测到物体（关闭时位置稳定且不在0）
    
    try:
        i = 0  # 循环计数器（秒）
        
        while True:
            time.sleep(1.0)  # 每秒循环一次
            i += 1
            
            # 如果检测到物体，停止打开操作
            if object_detected:
                print(f"[{i:3d}s] ⚠ 检测到物体，停止夹爪操作")
                print("  → 夹爪在关闭时位置稳定且不在0，说明已夹住物体")
                print()
                continue
            
            # 获取当前夹爪位置（用于判断是否在0）
            categorized_state = interface.get_joint_state(categorized=True)
            gripper_current_position = None
            if categorized_state:
                if is_dual_arm and 'left_gripper' in categorized_state:
                    gripper_data = categorized_state['left_gripper']
                    if gripper_data.get('positions'):
                        gripper_current_position = gripper_data['positions'][0]
                elif not is_dual_arm and 'gripper' in categorized_state:
                    gripper_data = categorized_state['gripper']
                    if gripper_data.get('positions'):
                        gripper_current_position = gripper_data['positions'][0]
            
            # 如果没有在等待到达，则发送下一个目标
            if not gripper_command_sent:
                gripper_target_open = not gripper_is_open
                gripper_position = 1.0 if gripper_target_open else 0.0
                gripper_status = "张开" if gripper_target_open else "闭合"
                
                try:
                    interface.send_gripper_command(gripper_position)
                    print(f"[{i:3d}s] → 夹爪: {gripper_status} (位置: {gripper_position:.1f})")
                    gripper_command_sent = True
                except Exception as e:
                    print(f"[{i:3d}s] ✗ 控制夹爪失败: {e}")
            else:
                # 检查是否到达
                try:
                    result = interface.check_arrive(gripper_part)
                    arrived = result.get('arrived', False)
                    distance = result.get('distance', float('inf'))
                    
                    if arrived:
                        # 检查是否在关闭时检测到物体（位置稳定且不在0）
                        if not gripper_target_open:  # 正在关闭
                            if gripper_current_position is not None and gripper_current_position > 0.0:
                                # 位置不在0，说明可能夹住物体了
                                # 检查位置历史是否稳定（单臂和双臂模式都使用 left_gripper_position_history）
                                history = interface.left_gripper_position_history
                                
                                if len(history) >= interface.gripper_stability_history_size:
                                    recent_positions = history[-interface.gripper_stability_history_size:]
                                    max_pos = max(recent_positions)
                                    min_pos = min(recent_positions)
                                    variance = max_pos - min_pos
                                    
                                    if variance < interface.gripper_stability_threshold:
                                        # 位置稳定且不在0，说明抓到物体了
                                        object_detected = True
                                        print(f"[{i:3d}s] ✓ 夹爪已到达关闭状态")
                                        print(f"  ⚠ 检测到物体！当前位置: {gripper_current_position:.4f} (不在0)")
                                        print(f"  → 位置稳定性: {variance:.4f} < {interface.gripper_stability_threshold:.4f}")
                                        print(f"  → 停止打开操作")
                                        print()
                                        continue
                        
                        # 正常到达
                        gripper_is_open = gripper_target_open
                        gripper_command_sent = False
                        print(f"[{i:3d}s] ✓ 夹爪已到达，下一步将切换状态")
                    else:
                        print(f"[{i:3d}s] ✗ 夹爪未到达 (距离: {distance:.4f})")
                except Exception as e:
                    print(f"[{i:3d}s] ⚠ 检查失败: {e}")
            
            print()
                
    except KeyboardInterrupt:
        # 用户中断（Ctrl+C）
        print("\n" + "-" * 70)
        print("  用户中断")
        print("-" * 70)
    
    # ========================================================================
    # 第四部分：断开连接和清理
    # ========================================================================
    print("\n" + "=" * 70)
    print("[9] 断开连接...")
    interface.disconnect()
    print("  ✓ 接口断开成功!")
    print("=" * 70)
    print("\n  测试完成!\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user - cleaning up...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
