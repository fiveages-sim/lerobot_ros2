"""
W2 机器人路径测试脚本
测试新的 target_path 接口，使用 w2轨迹测试.txt 中的轨迹数据
"""

import time
import sys
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header

from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig


def vector_to_pose(vector):
    """将7维向量 [x, y, z, qx, qy, qz, qw] 转换为 Pose 对象"""
    pose = Pose()
    pose.position.x = vector[0]
    pose.position.y = vector[1]
    pose.position.z = vector[2]
    pose.orientation.x = vector[3]
    pose.orientation.y = vector[4]
    pose.orientation.z = vector[5]
    pose.orientation.w = vector[6]
    return pose


def vectors_to_poses(vectors, frame_id="arm_base"):
    """将多个7维向量转换为 PoseStamped 列表"""
    poses = []
    for vec in vectors:
        pose_stamped = PoseStamped()
        pose_stamped.header = Header(frame_id=frame_id)
        pose_stamped.pose = vector_to_pose(vec)
        poses.append(pose_stamped)
    return poses


def main():
    """测试W2机器人的路径接口"""
    
    # ========================================================================
    # 配置参数
    # ========================================================================
    TRAJECTORY_EXECUTION_WAIT_TIME = 2.0  # 轨迹执行完成后的等待时间（秒）
    
    print("\n" + "=" * 70)
    print(" " * 20 + "W2 Robot Path Test")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # 第一部分：初始化和连接
    # ========================================================================
    print("[1] 创建配置...")
    config = ROS2RobotInterfaceConfig()
    
    print("[2] 创建ROS2RobotInterface实例...")
    interface = ROS2RobotInterface(config)
    
    print("[3] 连接到ROS 2...")
    try:
        interface.connect()
        print("    ✓ 接口连接成功!\n")
    except Exception as e:
        print(f"    ✗ 连接失败: {e}\n")
        return 1
    
    # 检查是否为双臂模式
    is_dual_arm = interface.config.right_end_effector_target_topic is not None
    if not is_dual_arm:
        print("    ✗ 错误: 此测试需要双臂模式，但未检测到右臂topic\n")
        interface.disconnect()
        return 1
    
    print("    ✓ 检测到双臂模式\n")
    
    # 等待数据到达
    print("[4] 等待数据到达（2秒）...")
    time.sleep(2.0)
    print("    ✓ 数据收集已开始\n")
    
    # ========================================================================
    # 第二部分：切换到OCS2状态
    # ========================================================================
    print("-" * 70)
    print("[5] 切换到OCS2状态")
    print("-" * 70)
    try:
        print("  → 切换到Hold状态...")
        interface.send_fsm_command(2)  # 2 = Hold状态
        time.sleep(1.0)
        print("  ✓ 已切换到Hold状态")
        # 先切换到HOME状态
        print("  → 切换到HOME状态...")
        interface.send_fsm_command(1)  # 1 = HOME状态
        time.sleep(5.0)
        print("  ✓ 已切换到HOME状态")
        
                # 先切换到HOME状态
        print("  → 再切换到Hold状态...")
        interface.send_fsm_command(2)  # 2 = Hold状态
        time.sleep(1.0)
        print("  ✓ 已切换到Hold状态")

        # 再切换到OCS2状态
        print("  → 切换到OCS2状态...")
        interface.send_fsm_command(3)  # 3 = OCS2/MOVE状态
        time.sleep(2.0)  # 等待状态转换完成
        print("  ✓ 已切换到OCS2状态\n")
    except Exception as e:
        print(f"  ✗ 切换到OCS2状态失败: {e}\n")
        interface.disconnect()
        return 1
    
    # ========================================================================
    # 第三部分：准备轨迹数据（来自 w2轨迹测试.txt）
    # ========================================================================
    print("-" * 70)
    print("[6] 准备轨迹数据")
    print("-" * 70)
    
    # A点轨迹数据
    # A点 - Left Arm（左臂）
    a_left_initial = [
        [0.243304, 0.287482, -0.166317, 0.669167, -0.144600, 0.714637, 0.143530],
        [0.464421, 0.188690, -0.078523, 0.669699, -0.139977, 0.715802, 0.139776]
    ]
    
    a_left_end = [
        [0.464421, 0.188690, -0.018523, 0.669699, -0.139977, 0.715802, 0.139776],
        [0.243304, 0.287482, -0.166317, 0.669167, -0.144600, 0.714637, 0.143530]
    ]
    
    # A点 - Right Arm（右臂）
    a_right_initial = [
        [0.243304, -0.287482, -0.166317, 0.669167, 0.144600, 0.714637, -0.143530],
        [0.464421, -0.188690, -0.078523, 0.669699, 0.139977, 0.715802, -0.139776]
    ]
    
    a_right_end = [
        [0.464421, -0.188690, -0.018523, 0.669699, 0.139977, 0.715802, -0.139776],
        [0.243304, -0.287482, -0.166317, 0.669167, 0.144600, 0.714637, -0.143530]
    ]
    
    # B点轨迹数据
    # B点 - Left Arm（左臂）
    b_left_initial = [
        [0.468027, 0.254521, -0.090424, 0.583926, 0.063293, 0.808647, 0.033391],
        [0.634115, 0.274643, 0.018583, 0.588553, 0.004895, 0.808427, -0.005291]
    ]
    
    b_left_end = [
        [0.655319, 0.276373, 0.036224, 0.700671, 0.022995, 0.713072, -0.007775],
        [0.654674, 0.331464, -0.028880, 0.485272, 0.506258, 0.515121, 0.492814],
        [0.385897, 0.444166, -0.046988, 0.502312, 0.483387, 0.521046, 0.492474]
    ]
    
    # B点 - Right Arm（右臂）
    b_right_initial = [
        [0.468027, -0.254521, -0.090424, 0.583926, -0.063293, 0.808647, -0.033391],
        [0.634115, -0.274643, 0.018583, 0.588553, -0.004895, 0.808427, 0.005291]
    ]
    
    b_right_end = [
        [0.655319, -0.276373, 0.036224, 0.700671, -0.022995, 0.713072, 0.007775],
        [0.654674, -0.311464, -0.028880, 0.485272, -0.506258, 0.515121, -0.492814],
        [0.385897, -0.444166, -0.046988, 0.502312, -0.483387, 0.521046, -0.492474]
    ]
    
    print("  ✓ 轨迹数据已加载")
    print(f"    A点: 左臂 {len(a_left_initial)} 个initial点, {len(a_left_end)} 个end点")
    print(f"    A点: 右臂 {len(a_right_initial)} 个initial点, {len(a_right_end)} 个end点")
    print(f"    B点: 左臂 {len(b_left_initial)} 个initial点, {len(b_left_end)} 个end点")
    print(f"    B点: 右臂 {len(b_right_initial)} 个initial点, {len(b_right_end)} 个end点\n")
    
    # ========================================================================
    # 第四部分：测试A点轨迹
    # ========================================================================
    print("=" * 70)
    print("[7] 测试A点轨迹 - Initial路径")
    print("=" * 70)
    
    try:
        # 转换A点initial轨迹
        a_left_initial_poses = vectors_to_poses(a_left_initial, frame_id="arm_base")
        a_right_initial_poses = vectors_to_poses(a_right_initial, frame_id="arm_base")
        
        print(f"  → 发送A点initial路径...")
        print(f"    左臂: {len(a_left_initial_poses)} 个路径点")
        print(f"    右臂: {len(a_right_initial_poses)} 个路径点")
        
        interface.send_target_path(a_left_initial_poses, a_right_initial_poses, frame_id="arm_base")
        print("  ✓ A点initial路径已发送")
        print(f"  → 等待轨迹执行完成（约{TRAJECTORY_EXECUTION_WAIT_TIME}秒）...")
        time.sleep(TRAJECTORY_EXECUTION_WAIT_TIME)
        print("  ✓ A点initial轨迹执行完成\n")
    except Exception as e:
        print(f"  ✗ 发送A点initial路径失败: {e}\n")
    
    # 等待一下
    time.sleep(1.0)
    
    print("-" * 70)
    print("[8] 测试A点轨迹 - End路径")
    print("-" * 70)
    
    try:
        # 转换A点end轨迹
        a_left_end_poses = vectors_to_poses(a_left_end, frame_id="arm_base")
        a_right_end_poses = vectors_to_poses(a_right_end, frame_id="arm_base")
        
        print(f"  → 发送A点end路径...")
        print(f"    左臂: {len(a_left_end_poses)} 个路径点")
        print(f"    右臂: {len(a_right_end_poses)} 个路径点")
        
        interface.send_target_path(a_left_end_poses, a_right_end_poses, frame_id="arm_base")
        print("  ✓ A点end路径已发送")
        print(f"  → 等待轨迹执行完成（约{TRAJECTORY_EXECUTION_WAIT_TIME}秒）...")
        time.sleep(TRAJECTORY_EXECUTION_WAIT_TIME)
        print("  ✓ A点end轨迹执行完成\n")
    except Exception as e:
        print(f"  ✗ 发送A点end路径失败: {e}\n")
    
    # ========================================================================
    # 第五部分：测试B点轨迹
    # ========================================================================
    time.sleep(2.0)
    
    print("=" * 70)
    print("[9] 测试B点轨迹 - Initial路径")
    print("=" * 70)
    
    try:
        # 转换B点initial轨迹
        b_left_initial_poses = vectors_to_poses(b_left_initial, frame_id="arm_base")
        b_right_initial_poses = vectors_to_poses(b_right_initial, frame_id="arm_base")
        
        print(f"  → 发送B点initial路径...")
        print(f"    左臂: {len(b_left_initial_poses)} 个路径点")
        print(f"    右臂: {len(b_right_initial_poses)} 个路径点")
        
        interface.send_target_path(b_left_initial_poses, b_right_initial_poses, frame_id="arm_base")
        print("  ✓ B点initial路径已发送")
        print(f"  → 等待轨迹执行完成（约{TRAJECTORY_EXECUTION_WAIT_TIME}秒）...")
        time.sleep(TRAJECTORY_EXECUTION_WAIT_TIME)
        print("  ✓ B点initial轨迹执行完成\n")
    except Exception as e:
        print(f"  ✗ 发送B点initial路径失败: {e}\n")
    
    # 等待一下
    time.sleep(1.0)
    
    print("-" * 70)
    print("[10] 测试B点轨迹 - End路径")
    print("-" * 70)
    
    try:
        # 转换B点end轨迹
        b_left_end_poses = vectors_to_poses(b_left_end, frame_id="arm_base")
        b_right_end_poses = vectors_to_poses(b_right_end, frame_id="arm_base")
        
        print(f"  → 发送B点end路径...")
        print(f"    左臂: {len(b_left_end_poses)} 个路径点")
        print(f"    右臂: {len(b_right_end_poses)} 个路径点")
        
        interface.send_target_path(b_left_end_poses, b_right_end_poses, frame_id="arm_base")
        print("  ✓ B点end路径已发送")
        print(f"  → 等待轨迹执行完成（约{TRAJECTORY_EXECUTION_WAIT_TIME}秒）...")
        time.sleep(TRAJECTORY_EXECUTION_WAIT_TIME)
        print("  ✓ B点end轨迹执行完成\n")
    except Exception as e:
        print(f"  ✗ 发送B点end路径失败: {e}\n")
    
    # ========================================================================
    # 第六部分：测试A点initial第一个点（单点轨迹）
    # ========================================================================
    time.sleep(2.0)
    
    print("=" * 70)
    print("[11] 测试A点initial第一个点 - 单点轨迹")
    print("=" * 70)
    
    try:
        # 取A点initial的第一个点
        a_left_initial_first = [a_left_initial[0]]  # 只取第一个点
        a_right_initial_first = [a_right_initial[0]]  # 只取第一个点
        
        # 转换为PoseStamped
        a_left_initial_first_poses = vectors_to_poses(a_left_initial_first, frame_id="arm_base")
        a_right_initial_first_poses = vectors_to_poses(a_right_initial_first, frame_id="arm_base")
        
        print(f"  → 发送A点initial第一个点路径...")
        print(f"    左臂: {len(a_left_initial_first_poses)} 个路径点")
        print(f"    右臂: {len(a_right_initial_first_poses)} 个路径点")
        
        interface.send_target_path(a_left_initial_first_poses, a_right_initial_first_poses, frame_id="arm_base")
        print("  ✓ A点initial第一个点路径已发送")
        print(f"  → 等待轨迹执行完成（约{TRAJECTORY_EXECUTION_WAIT_TIME}秒）...")
        time.sleep(TRAJECTORY_EXECUTION_WAIT_TIME)
        print("  ✓ A点initial第一个点轨迹执行完成\n")
    except Exception as e:
        print(f"  ✗ 发送A点initial第一个点路径失败: {e}\n")
    
    # ========================================================================
    # 第七部分：完成和清理
    # ========================================================================
    print("=" * 70)
    print("[12] 测试完成")
    print("=" * 70)
    print("  ✓ 所有轨迹测试已完成")
    print("  → 等待3秒后断开连接...\n")
    time.sleep(3.0)
    
    try:
        interface.disconnect()
        print("  ✓ 已断开连接\n")
    except Exception as e:
        print(f"  ⚠ 断开连接时出错: {e}\n")
    
    print("=" * 70)
    print("测试结束")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

