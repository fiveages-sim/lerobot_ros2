"""
ROS2机器人接口测试脚本
支持单臂和双臂机器人的通用测试
"""

import time
import sys
from geometry_msgs.msg import Pose

from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig, ControlType


def main():
    """测试ROS2机器人接口（支持单臂和双臂机器人）"""
    
    # ========================================================================
    # 第一部分：初始化和连接
    # ========================================================================
    print("\n" + "=" * 70)
    print(" " * 20 + "ROS2 Robot Interface Test")
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
    # 第三部分：FSM状态循环配置
    # ========================================================================
    print("-" * 70)
    print("[8] FSM状态循环配置")
    print("-" * 70)
    # Define FSM state cycle: HOLD → HOME → REST → REST → HOLD → OCS2 → HOLD → MOVEJ (then loop back to HOLD)
    # FSM command values: 0=SWITCH, 1=HOME, 2=HOLD, 3=OCS2/MOVE, 4=MOVEJ, 100=REST
    fsm_states = [2, 1, 100, 100, 100, 100, 2, 3, 2, 4]  # HOLD → HOME → REST → REST → HOLD → OCS2 → HOLD → MOVEJ → (loop back to HOLD)
    fsm_state_names = {0: "SWITCH", 100: "REST", 1: "HOME", 2: "HOLD", 3: "OCS2/MOVE", 4: "MOVEJ"}
    # Start from index 1 (HOME state) since we just switched to it in step [5]
    current_fsm_index = 1  # Index 1 corresponds to HOME (state 1)
    print(f"  → FSM states cycle: {' → '.join([f'{s}({fsm_state_names[s]})' for s in fsm_states])} → (loop)")
    print(f"  → Switching every 5 seconds")
    print(f"  → Current state: {fsm_states[current_fsm_index]} ({fsm_state_names[fsm_states[current_fsm_index]]})")
    print(f"  → Next switch will go to: {fsm_states[(current_fsm_index + 1) % len(fsm_states)]} ({fsm_state_names[fsm_states[(current_fsm_index + 1) % len(fsm_states)]]})\n")
    
    # ========================================================================
    # 第四部分：主循环 - 持续监控数据和循环FSM状态
    # ========================================================================
    print("=" * 70)
    print("[9] 持续监控数据并循环FSM状态")
    print("=" * 70)
    is_dual_arm = interface.config.right_end_effector_pose_topic is not None
    if is_dual_arm:
        print("  模式: 双臂")
        print("  → 在OCS2状态: 双臂将依次伸出和收回")
        print("  → 在MOVEJ状态: 头部和身体将移动")
    else:
        print("  模式: 单臂")
    print("  → FSM状态每3秒切换一次")
    print("  → 夹爪基于到达判断切换状态（等待到达后自动切换）")
    print("  → 按Ctrl+C停止并断开连接")
    print("-" * 70 + "\n")
    
    # ========================================================================
    # 第五部分：动作控制变量初始化
    # ========================================================================
    
    # 手臂动作控制变量（用于OCS2状态）
    # 状态: idle, left_extending, left_extended, left_retracting, 
    #      right_extending, right_extended, right_retracting, completed
    arm_movement_state = "idle"  # 当前手臂动作状态
    arm_movement_start_time = None  # 动作开始时间
    left_initial_pose = None  # 左臂初始位姿
    right_initial_pose = None  # 右臂初始位姿
    extend_distance = 0.15  # 向前伸出的距离（米）
    movement_duration = 2.0  # 每个动作阶段的持续时间（秒）
    last_fsm_switch_time = 0  # 记录上次FSM状态切换的时间
    hold_state_enter_time = None  # 记录进入HOLD状态的时间
    
    # 头部和身体动作控制变量（用于MOVEJ状态）
    # 状态: idle, head_moving_left, head_moving_right, head_moving_up, 
    #      head_moving_down, body_moving_to_target, body_moving_back, completed
    head_body_movement_state = "idle"  # 当前头部/身体动作状态
    head_body_movement_start_time = None  # 动作开始时间
    head_initial_positions = None  # 头部关节初始位置
    body_initial_positions = None  # 身体关节初始位置
    # 注意：head_target_positions 和 body_target_positions 现在由 interface 自动管理
    # 在调用 interface.send_head_joint_positions() 或 interface.send_body_joint_positions() 时会自动设置
    
    # 夹爪控制变量（每个夹爪独立管理）
    left_gripper_is_open = False  # 当前左夹爪状态：False=闭合(0.0), True=张开(1.0)
    left_gripper_target_open = True  # 下一个目标状态
    left_gripper_command_sent = False  # 左夹爪是否已发送命令，等待到达
    left_gripper_arrived = False  # 左夹爪是否到达目标

    right_gripper_is_open = False  # 右夹爪当前状态（双臂模式）
    right_gripper_target_open = True  # 右夹爪下一个目标状态
    right_gripper_command_sent = False  # 右夹爪是否已发送命令，等待到达
    right_gripper_arrived = False  # 右夹爪是否到达目标
    
    try:
        i = 0  # 循环计数器（秒）
        current_fsm_state = fsm_states[current_fsm_index]  # 跟踪当前FSM状态
        
        # ========================================================================
        # 第六部分：主循环 - 持续监控和控制
        # ========================================================================
        while True:
            
            time.sleep(1.0)  # 每秒循环一次
            i += 1
            
            # 获取当前状态数据
            joint_state = interface.get_joint_state()  # 获取关节状态
            left_pose = interface.get_end_effector_pose()  # 获取左臂末端位姿
            right_pose = interface.get_right_end_effector_pose()  # 获取右臂末端位姿（双臂模式）
            
            # ========================================================================
            # 第六点五部分：夹爪开合测试（基于到达判断切换，左右独立）
            # ========================================================================
            # 左夹爪：如果没有在等待到达，则发送下一个目标；如果在等待，则检查到达
            if not left_gripper_command_sent:
                left_gripper_target_open = not left_gripper_is_open
                left_position = 1.0 if left_gripper_target_open else 0.0
                left_status = "张开" if left_gripper_target_open else "闭合"
                try:
                    interface.send_gripper_command(left_position)
                    print(f"  [夹爪] → 左臂夹爪: {left_status} (位置: {left_position:.1f})")
                    left_gripper_command_sent = True
                    left_gripper_arrived = False
                except Exception as e:
                    print(f"  [夹爪] ✗ 控制左臂夹爪失败: {e}")
            else:
                try:
                    left_res = interface.check_arrive('left_gripper' if is_dual_arm else 'gripper')
                    left_gripper_arrived = left_res.get('arrived', False)
                except Exception:
                    left_gripper_arrived = True
                if left_gripper_arrived:
                    left_gripper_is_open = left_gripper_target_open
                    left_gripper_command_sent = False
                    print(f"  [夹爪] ✓ 左臂夹爪已到达，下一步将切换状态")

            # 右夹爪：仅双臂模式时独立处理
            if is_dual_arm:
                if not right_gripper_command_sent:
                    right_gripper_target_open = not right_gripper_is_open
                    right_position = 1.0 if right_gripper_target_open else 0.0
                    right_status = "张开" if right_gripper_target_open else "闭合"
                    try:
                        interface.send_right_gripper_command(right_position)
                        print(f"  [夹爪] → 右臂夹爪: {right_status} (位置: {right_position:.1f})")
                        right_gripper_command_sent = True
                        right_gripper_arrived = False
                    except Exception as e:
                        print(f"  [夹爪] ✗ 控制右臂夹爪失败: {e}")
                else:
                    try:
                        right_res = interface.check_arrive('right_gripper')
                        right_gripper_arrived = right_res.get('arrived', False)
                    except Exception:
                        right_gripper_arrived = True
                    if right_gripper_arrived:
                        right_gripper_is_open = right_gripper_target_open
                        right_gripper_command_sent = False
                        print(f"  [夹爪] ✓ 右臂夹爪已到达，下一步将切换状态")
            
            # ========================================================================
            # 第七部分：OCS2状态下的双臂协调控制
            # ========================================================================
            # 在OCS2状态下控制双臂依次伸出和收回（仅双臂模式）
            if current_fsm_state == 3 and is_dual_arm:  # OCS2状态
                current_time = time.time()
                
                # 初始化：保存初始位姿（仅在idle状态且获取到位姿时执行一次）
                if arm_movement_state == "idle" and left_pose and right_pose:
                    left_initial_pose = Pose()
                    left_initial_pose.position.x = left_pose.position.x
                    left_initial_pose.position.y = left_pose.position.y
                    left_initial_pose.position.z = left_pose.position.z
                    left_initial_pose.orientation.x = left_pose.orientation.x
                    left_initial_pose.orientation.y = left_pose.orientation.y
                    left_initial_pose.orientation.z = left_pose.orientation.z
                    left_initial_pose.orientation.w = left_pose.orientation.w
                    
                    right_initial_pose = Pose()
                    right_initial_pose.position.x = right_pose.position.x
                    right_initial_pose.position.y = right_pose.position.y
                    right_initial_pose.position.z = right_pose.position.z
                    right_initial_pose.orientation.x = right_pose.orientation.x
                    right_initial_pose.orientation.y = right_pose.orientation.y
                    right_initial_pose.orientation.z = right_pose.orientation.z
                    right_initial_pose.orientation.w = right_pose.orientation.w
                    
                    # 开始左臂伸出动作
                    arm_movement_state = "left_extending"
                    arm_movement_start_time = current_time
                    # 发送左臂伸出命令：使用相对坐标系，X方向向前伸出 extend_distance
                    left_target = Pose()
                    left_target.position.x = extend_distance  # 相对于 left_eef 坐标系向前
                    left_target.position.y = 0.0  # 相对于 left_eef 坐标系
                    left_target.position.z = 0.0  # 相对于 left_eef 坐标系
                    left_target.orientation.w = 1.0  # 保持初始姿态（单位四元数）
                    left_target.orientation.x = 0.0
                    left_target.orientation.y = 0.0
                    left_target.orientation.z = 0.0
                    interface.send_end_effector_target_stamped("left_eef", left_target)
                    # # 旧方法：使用绝对坐标系
                    # left_target = Pose()
                    # left_target.position.x = left_initial_pose.position.x + extend_distance  # 向前伸出
                    # left_target.position.y = left_initial_pose.position.y
                    # left_target.position.z = left_initial_pose.position.z
                    # left_target.orientation = left_initial_pose.orientation  # 保持初始姿态
                    # interface.send_end_effector_target(left_target)
                    print(f"  [OCS2] → 左臂向前伸出（向前 {extend_distance}m）...")
                
                # 状态机：左臂伸出中
                elif arm_movement_state == "left_extending":
                    # 检查左臂是否到达目标位置，或超时后切换
                    left_arm_check = interface.check_arrive('left_arm')
                    if left_arm_check['arrived']:
                        arm_movement_state = "left_extended"
                        arm_movement_start_time = current_time
                       
                
                # 状态机：左臂已伸出，等待1秒后开始收回
                elif arm_movement_state == "left_extended":
                    if current_time - arm_movement_start_time >= 1.0:  # 等待1秒
                        arm_movement_state = "left_retracting"
                        arm_movement_start_time = current_time
                        # 发送左臂收回命令：使用相对坐标系，向后移动 extend_distance
                        left_retract = Pose()
                        left_retract.position.x = -extend_distance  # 相对于 left_eef 坐标系向后
                        left_retract.position.y = 0.0
                        left_retract.position.z = 0.0
                        left_retract.orientation.w = 1.0  # 保持初始姿态（单位四元数）
                        left_retract.orientation.x = 0.0
                        left_retract.orientation.y = 0.0
                        left_retract.orientation.z = 0.0
                        interface.send_end_effector_target_stamped("left_eef", left_retract)
                        # # 旧方法：使用绝对坐标系
                        # if left_initial_pose:
                        #     left_retract = Pose()
                        #     left_retract.position.x = left_initial_pose.position.x - extend_distance  # 向后收回
                        #     left_retract.position.y = left_initial_pose.position.y
                        #     left_retract.position.z = left_initial_pose.position.z
                        #     left_retract.orientation = left_initial_pose.orientation  # 保持初始姿态
                        #     interface.send_end_effector_target(left_retract)
                        print(f"  [OCS2] → 左臂收回中（向后 {extend_distance}m）...")
                
                # 状态机：左臂收回中
                elif arm_movement_state == "left_retracting":
                    # 检查左臂是否到达目标位置，或超时后切换
                    left_arm_check = interface.check_arrive('left_arm')
                    if left_arm_check['arrived']:
                        # 开始右臂伸出
                        arm_movement_state = "right_extending"
                        arm_movement_start_time = current_time
                    
                        # 发送右臂伸出命令：使用相对坐标系，X方向向前伸出 extend_distance
                        right_target = Pose()
                        right_target.position.x = extend_distance  # 相对于 right_eef 坐标系向前
                        right_target.position.y = 0.0  # 相对于 right_eef 坐标系
                        right_target.position.z = 0.0  # 相对于 right_eef 坐标系
                        right_target.orientation.w = 1.0  # 保持初始姿态（单位四元数）
                        right_target.orientation.x = 0.0
                        right_target.orientation.y = 0.0
                        right_target.orientation.z = 0.0
                        interface.send_right_end_effector_target_stamped("right_eef", right_target)
                        # # 旧方法：使用绝对坐标系
                        # if right_initial_pose:
                        #     right_target = Pose()
                        #     right_target.position.x = right_initial_pose.position.x + extend_distance  # 向前伸出
                        #     right_target.position.y = right_initial_pose.position.y
                        #     right_target.position.z = right_initial_pose.position.z
                        #     right_target.orientation = right_initial_pose.orientation  # 保持初始姿态
                        #     interface.send_right_end_effector_target(right_target)
                        print(f"  [OCS2] → 右臂向前伸出（向前 {extend_distance}m）...")
                
                # 状态机：右臂伸出中
                elif arm_movement_state == "right_extending":
                    # 检查右臂是否到达目标位置，或超时后切换
                    right_arm_check = interface.check_arrive('right_arm')
                    if right_arm_check['arrived']:
                        arm_movement_state = "right_extended"
                        arm_movement_start_time = current_time
                      
                
                # 状态机：右臂已伸出，等待1秒后开始收回
                elif arm_movement_state == "right_extended":
                    if current_time - arm_movement_start_time >= 1.0:  # 等待1秒
                        arm_movement_state = "right_retracting"
                        arm_movement_start_time = current_time
                        # 发送右臂收回命令：使用相对坐标系，向后移动 extend_distance
                        right_retract = Pose()
                        right_retract.position.x = -extend_distance  # 相对于 right_eef 坐标系向后
                        right_retract.position.y = 0.0
                        right_retract.position.z = 0.0
                        right_retract.orientation.w = 1.0  # 保持初始姿态（单位四元数）
                        right_retract.orientation.x = 0.0
                        right_retract.orientation.y = 0.0
                        right_retract.orientation.z = 0.0
                        interface.send_right_end_effector_target_stamped("right_eef", right_retract)
                        # # 旧方法：使用绝对坐标系
                        # if right_initial_pose:
                        #     right_retract = Pose()
                        #     right_retract.position.x = right_initial_pose.position.x - extend_distance  # 向后收回
                        #     right_retract.position.y = right_initial_pose.position.y
                        #     right_retract.position.z = right_initial_pose.position.z
                        #     right_retract.orientation = right_initial_pose.orientation  # 保持初始姿态
                        #     interface.send_right_end_effector_target(right_retract)
                        print(f"  [OCS2] → 右臂收回中（向后 {extend_distance}m）...")
                
                # 状态机：右臂收回中
                elif arm_movement_state == "right_retracting":
                    # 检查右臂是否到达目标位置，或超时后切换
                    right_arm_check = interface.check_arrive('right_arm')
                    if right_arm_check['arrived']:
                        arm_movement_state = "completed"  # 标记为完成，准备切换状态
                        arm_movement_start_time = None
                       
            
            # ========================================================================
            # 第八部分：MOVEJ状态下的头部和身体关节控制
            # ========================================================================
            # 在MOVEJ状态下控制头部和身体关节
            elif current_fsm_state == 4:  # MOVEJ状态
                current_time = time.time()
                
                # 初始化：保存头部和身体关节的初始位置（仅在idle状态时执行一次）
                if head_body_movement_state == "idle":
                    categorized_state = interface.get_joint_state(categorized=True)
                    if categorized_state:
                        if categorized_state.get('head', {}).get('positions'):
                            head_initial_positions = categorized_state['head']['positions'].copy()
                        if categorized_state.get('body', {}).get('positions'):
                            body_initial_positions = categorized_state['body']['positions'].copy()
                    
                    # 开始头部动作：先左右摆动，再上下摆动
                    head_body_movement_state = "head_moving_left"
                    head_body_movement_start_time = current_time
                    
                    # 头部向左摆动：head_joint1（索引0）控制左右（yaw），增加0.3弧度
                    if head_initial_positions and interface.config.head_joint_controller_topic:
                        head_target_left = head_initial_positions.copy()
                        if len(head_target_left) > 0:
                            head_target_left[0] = head_initial_positions[0] + 0.3  # 向左摆动（yaw增加）
                        # 目标位置已由 interface.send_head_joint_positions() 自动记录
                        try:
                            interface.send_head_joint_positions(head_target_left)
                            print(f"  [MOVEJ] → 头部向左摆动...")
                        except Exception as e:
                            print(f"  [MOVEJ] ✗ 控制头部失败: {e}")
                
                # 状态机：头部向左摆动中
                elif head_body_movement_state == "head_moving_left":
                    # 检查是否到达目标位置，或超时后切换
                    head_check = interface.check_arrive('head')
                    if head_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "head_moving_right"
                        head_body_movement_start_time = current_time
                        # 头部向右摆动：head_joint1（索引0）减少0.3弧度
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            head_target_right = head_initial_positions.copy()
                            if len(head_target_right) > 0:
                                head_target_right[0] = head_initial_positions[0] - 0.3  # 向右摆动（yaw减少）
                            # 目标位置已由 interface.send_head_joint_positions() 自动记录
                            try:
                                interface.send_head_joint_positions(head_target_right)
                                print(f"  [MOVEJ] → 头部向右摆动...")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制头部失败: {e}")
                
                # 状态机：头部向右摆动中
                elif head_body_movement_state == "head_moving_right":
                    # 检查是否到达目标位置，或超时后切换
                    head_check = interface.check_arrive('head')
                    if head_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "head_moving_up"
                        head_body_movement_start_time = current_time
                        # 头部向上摆动：head_joint2（索引1）控制上下（pitch），减少0.3弧度（pitch减少=向上）
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            head_target_up = head_initial_positions.copy()
                            if len(head_target_up) > 1:
                                head_target_up[1] = head_initial_positions[1] - 0.3  # 向上摆动（pitch减少）
                            # 目标位置已由 interface.send_head_joint_positions() 自动记录
                            try:
                                interface.send_head_joint_positions(head_target_up)
                                print(f"  [MOVEJ] → 头部向上摆动...")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制头部失败: {e}")
                
                # 状态机：头部向上摆动中
                elif head_body_movement_state == "head_moving_up":
                    # 检查是否到达目标位置，或超时后切换
                    head_check = interface.check_arrive('head')
                    if head_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "head_moving_down"
                        head_body_movement_start_time = current_time
                        # 头部向下摆动：head_joint2（索引1）增加0.3弧度（pitch增加=向下）
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            head_target_down = head_initial_positions.copy()
                            if len(head_target_down) > 1:
                                head_target_down[1] = head_initial_positions[1] + 0.3  # 向下摆动（pitch增加）
                            # 目标位置已由 interface.send_head_joint_positions() 自动记录
                            try:
                                interface.send_head_joint_positions(head_target_down)
                                print(f"  [MOVEJ] → 头部向下摆动...")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制头部失败: {e}")
                
                # 状态机：头部向下摆动中
                elif head_body_movement_state == "head_moving_down":
                    # 检查是否到达目标位置，或超时后切换
                    head_check = interface.check_arrive('head')
                    if head_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "body_moving_to_target"
                        head_body_movement_start_time = current_time
                        # 头部回到初始位置
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            # 目标位置已由 interface.send_head_joint_positions() 自动记录
                            try:
                                interface.send_head_joint_positions(head_initial_positions)
                                print(f"  [MOVEJ] → 头部回到初始位置")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制头部失败: {e}")
                        
                        # 开始身体动作：移动到目标位置
                        if body_initial_positions and interface.config.body_joint_controller_topic:
                            # 身体目标位置（可以在这里手动修改目标位置值）
                            body_target_fixed = [-0.899, -1.714, -0.865, 0.0]  # 身体关节目标位置（弧度）
                            
                            # 确保目标位置数组长度与当前位置数组长度一致
                            body_target = body_target_fixed.copy()
                            if len(body_target) < len(body_initial_positions):
                                # 如果目标数组较短，用初始位置填充
                                body_target.extend(body_initial_positions[len(body_target):])
                            elif len(body_target) > len(body_initial_positions):
                                # 如果目标数组较长，截断到当前长度
                                body_target = body_target[:len(body_initial_positions)]
                            
                            # 目标位置已由 interface.send_body_joint_positions() 自动记录
                            try:
                                interface.send_body_joint_positions(body_target)
                                print(f"  [MOVEJ] → 身体移动到目标位置: {[f'{p:.3f}' for p in body_target]}")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制身体失败: {e}")
                
                # 状态机：身体移动到目标位置中
                elif head_body_movement_state == "body_moving_to_target":
                    # 检查是否到达目标位置，或超时后切换
                    body_check = interface.check_arrive('body')
                    if body_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "body_moving_back"
                        head_body_movement_start_time = current_time
                        # 身体回到初始位置
                        if body_initial_positions and interface.config.body_joint_controller_topic:
                            # 目标位置已由 interface.send_body_joint_positions() 自动记录
                            try:
                                interface.send_body_joint_positions(body_initial_positions)
                                print(f"  [MOVEJ] → 身体回到初始位置: {[f'{p:.3f}' for p in body_initial_positions]}")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ 控制身体失败: {e}")
                
                # 状态机：身体返回初始位置中
                elif head_body_movement_state == "body_moving_back":
                    # 检查是否到达目标位置，或超时后切换
                    body_check = interface.check_arrive('body')
                    if body_check['arrived'] or (current_time - head_body_movement_start_time >= movement_duration):
                        head_body_movement_state = "completed"  # 标记为完成，准备切换状态
                        head_body_movement_start_time = None
                        head_initial_positions = None
                        body_initial_positions = None
                        # 目标位置由 interface 自动管理，无需手动清除
                        print(f"  [MOVEJ] → 所有动作完成（头部、身体），准备切换状态...")
            
            
            # ========================================================================
            # 第十部分：状态显示和监控
            # ========================================================================
            
            # # 每秒显示一次状态信息
            # status_line = f"[{i:3d}s] "
            
            # # 显示关节状态（简化版）
            # if joint_state and len(joint_state['positions']) > 0:
            #     first_joint_pos = joint_state['positions'][0] if joint_state['positions'] else 0.0
            #     status_line += f"Joints: {len(joint_state['names'])} | "
            
            # # 显示末端执行器位姿
            # if left_pose:
            #     status_line += f"Left: ({left_pose.position.x:6.3f}, {left_pose.position.y:6.3f}, {left_pose.position.z:6.3f})"
            # if right_pose:
            #     status_line += f" | Right: ({right_pose.position.x:6.3f}, {right_pose.position.y:6.3f}, {right_pose.position.z:6.3f})"
            
            # print(status_line)
            
            # # 每5秒显示一次详细的分类关节状态
            # if i % 5 == 0:
            #     categorized_joint_state = interface.get_joint_state(categorized=True)
            #     if categorized_joint_state:
            #         is_dual_arm = interface.config.right_end_effector_pose_topic is not None
            #         print()
            #         print(f"  [{i}s] Categorized Joint States:")
            #         print("-" * 70)
                    
            #         if is_dual_arm:
            #             # Dual-arm mode
            #             if categorized_joint_state.get('left_arm', {}).get('names'):
            #                 left_arm = categorized_joint_state['left_arm']
            #                 print(f"  Left Arm ({len(left_arm['names'])} joints):")
            #                 for j, name in enumerate(left_arm['names']):
            #                     pos = left_arm['positions'][j] if j < len(left_arm['positions']) else 0.0
            #                     vel = left_arm['velocities'][j] if j < len(left_arm['velocities']) else 0.0
            #                     print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                        
            #             if categorized_joint_state.get('right_arm', {}).get('names'):
            #                 right_arm = categorized_joint_state['right_arm']
            #                 print(f"  Right Arm ({len(right_arm['names'])} joints):")
            #                 for j, name in enumerate(right_arm['names']):
            #                     pos = right_arm['positions'][j] if j < len(right_arm['positions']) else 0.0
            #                     vel = right_arm['velocities'][j] if j < len(right_arm['velocities']) else 0.0
            #                     print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
            #         else:
            #             # Single-arm mode
            #             if categorized_joint_state.get('arm', {}).get('names'):
            #                 arm = categorized_joint_state['arm']
            #                 print(f"  Arm ({len(arm['names'])} joints):")
            #                 for j, name in enumerate(arm['names']):
            #                     pos = arm['positions'][j] if j < len(arm['positions']) else 0.0
            #                     vel = arm['velocities'][j] if j < len(arm['velocities']) else 0.0
            #                     print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
            #         if categorized_joint_state.get('head', {}).get('names'):
            #             head = categorized_joint_state['head']
            #             print(f"  Head ({len(head['names'])} joints):")
            #             for j, name in enumerate(head['names']):
            #                 pos = head['positions'][j] if j < len(head['positions']) else 0.0
            #                 vel = head['velocities'][j] if j < len(head['velocities']) else 0.0
            #                 print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
            #         if categorized_joint_state.get('body', {}).get('names'):
            #             body = categorized_joint_state['body']
            #             print(f"  Body ({len(body['names'])} joints):")
            #             for j, name in enumerate(body['names']):
            #                 pos = body['positions'][j] if j < len(body['positions']) else 0.0
            #                 vel = body['velocities'][j] if j < len(body['velocities']) else 0.0
            #                 print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
            #         if categorized_joint_state.get('other', {}).get('names'):
            #             other = categorized_joint_state['other']
            #             print(f"  Other ({len(other['names'])} joints):")
            #             for j, name in enumerate(other['names']):
            #                 pos = other['positions'][j] if j < len(other['positions']) else 0.0
            #                 vel = other['velocities'][j] if j < len(other['velocities']) else 0.0
            #                 print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
            #         print("-" * 70)
            #         print()
            
            # ========================================================================
            # 第十一部分：FSM状态切换逻辑
            # ========================================================================
            # 检查是否应该切换FSM状态
            # HOLD状态：1秒后切换
            # 其他状态：每5秒切换一次，但如果OCS2/MOVEJ状态下的动作未完成，则等待动作完成
            # 动作完成后立即切换（不等待5秒周期）
            interface.check_arrive()
            should_switch_state = False
            switch_reason = ""
            
            # 跟踪进入HOLD状态的时间
            if current_fsm_state == 2:  # HOLD状态
                if hold_state_enter_time is None:
                    hold_state_enter_time = i  # 记录进入HOLD状态的时间
            else:
                hold_state_enter_time = None  # 离开HOLD状态时重置
            
            # 检查HOLD状态是否已停留1秒
            if current_fsm_state == 2 and hold_state_enter_time is not None:
                if i - hold_state_enter_time >= 1 and i != last_fsm_switch_time:
                    should_switch_state = True
                    switch_reason = "HOLD状态1秒"
            
            # 检查是否到了切换时间（每5秒，非HOLD状态）
            elif i % 3 == 0 and i > 0 and i != last_fsm_switch_time:
                # 如果在OCS2状态且动作未完成，延迟切换
                if current_fsm_state == 3 and is_dual_arm and arm_movement_state != "completed":
                    # print(f"  [8] ⏳ 等待手臂动作完成后再切换状态...")
                    # print(f"      当前动作状态: {arm_movement_state}")
                    pass
                # 如果在MOVEJ状态且动作未完成，延迟切换
                elif current_fsm_state == 4 and head_body_movement_state != "completed":
                    # print(f"  [8] ⏳ 等待头部/身体动作完成后再切换状态...")
                    # print(f"      当前动作状态: {head_body_movement_state}")
                    pass
                else:
                    should_switch_state = True
                    switch_reason = "3秒周期"
            
            # OCS2动作完成后立即切换（即使不在5秒周期）
            if current_fsm_state == 3 and is_dual_arm and arm_movement_state == "completed" and i != last_fsm_switch_time:
                should_switch_state = True
                switch_reason = "动作完成"
            
            # MOVEJ动作完成后立即切换（即使不在5秒周期）
            if current_fsm_state == 4 and head_body_movement_state == "completed" and i != last_fsm_switch_time:
                should_switch_state = True
                switch_reason = "动作完成"
            
            # 执行状态切换（如果满足条件）
            if should_switch_state:
                print()
                if switch_reason == "动作完成":
                    print(f"  [8] → 动作完成，立即切换状态...")
                # 移动到下一个FSM状态（循环）
                current_fsm_index = (current_fsm_index + 1) % len(fsm_states)
                next_fsm_state = fsm_states[current_fsm_index]
                next_fsm_name = fsm_state_names[next_fsm_state]
                
                # 检查FSM状态是否改变，如果离开OCS2或MOVEJ状态则重置动作状态
                if current_fsm_state == 3 and next_fsm_state != 3:
                    arm_movement_state = "idle"
                    arm_movement_start_time = None
                    left_initial_pose = None
                    right_initial_pose = None
                
                if current_fsm_state == 4 and next_fsm_state != 4:
                    head_body_movement_state = "idle"
                    head_body_movement_start_time = None
                    head_initial_positions = None
                    body_initial_positions = None
                
                # 如果进入HOLD状态，重置hold_state_enter_time（会在下次循环中设置）
                if next_fsm_state == 2:
                    hold_state_enter_time = None
                
                try:
                    interface.send_fsm_command(next_fsm_state)
                    print(f"  [8] ✓ FSM状态已切换到: {next_fsm_state} ({next_fsm_name})")
                    
                    # 特殊处理：当状态为100(REST)时，发送完100后立即发送0(SWITCH)
                    if next_fsm_state == 100:
                        time.sleep(0.1)  # 短暂延迟，确保100命令已处理
                        interface.send_fsm_command(0)
                        print(f"  [8] ✓ FSM状态已切换到: 0 (SWITCH)")
                    
                    # 更新跟踪的FSM状态
                    current_fsm_state = next_fsm_state
                    last_fsm_switch_time = i  # 记录切换时间
                except Exception as e:
                    print(f"  [8] ✗ 切换FSM状态失败: {e}")
                print()
                
    except KeyboardInterrupt:
        # 用户中断（Ctrl+C）
        print("\n" + "-" * 70)
        print("  用户中断")
        print("-" * 70)
    
    # ========================================================================
    # 第十二部分：断开连接和清理
    # ========================================================================
    print("\n" + "=" * 70)
    print("[10] 断开连接...")
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
