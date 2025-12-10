import time
import sys
from geometry_msgs.msg import Pose

from ros2_robot_interface import ROS2RobotInterface, ROS2RobotInterfaceConfig, ControlType


def main():
    """Test ROS2 Robot Interface (supports both single-arm and dual-arm robots)."""
    print("\n" + "=" * 70)
    print(" " * 20 + "ROS2 Robot Interface Test")
    print("=" * 70 + "\n")
    
    # Create configuration (mode will be auto-detected from ROS 2 topics)
    print("[1] Creating configuration...")
    print("    → Mode will be auto-detected from ROS 2 topics")
    print("    → Checking for /right_target or /right_current_pose\n")
    config = ROS2RobotInterfaceConfig()
    
    # Create and connect interface
    print("[2] Creating ROS2RobotInterface instance...")
    interface = ROS2RobotInterface(config)
    
    print("[3] Connecting to ROS 2...")
    try:
        interface.connect()
        print("    ✓ Interface connected successfully!\n")
    except Exception as e:
        print(f"    ✗ Failed to connect: {e}\n")
        return 1
    
    # Wait a bit for data to arrive
    print("[4] Waiting for data (2 seconds)...")
    time.sleep(2.0)
    print("    ✓ Data collection started\n")
    
    # Switch to HOLD state (starting state for FSM cycle)
    print("-" * 70)
    print("[5] Switching to HOLD state (starting state)")
    print("-" * 70)
    try:
        interface.send_fsm_command(2)  # 2 = HOLD state
        print("  ✓ FSM command sent: Switching to HOLD state")
        time.sleep(1.0)  # Wait a bit for state transition
        print("  ✓ State transition completed\n")
    except Exception as e:
        print(f"  ⚠ Failed to switch to HOLD state: {e}\n")
    
    # Test: Get joint state (shared for both arms)
    print("-" * 70)
    print("[6] Testing get_joint_state()")
    print("-" * 70)
    joint_state = interface.get_joint_state()
    if joint_state:
        print(f"  ✓ Joint state received")
        print(f"    Total joints: {len(joint_state['names'])}")
        print(f"    Joint names: {', '.join(joint_state['names'][:5])}{'...' if len(joint_state['names']) > 5 else ''}")
        print(f"    Positions:   {[f'{p:.3f}' for p in joint_state['positions'][:5]]}{'...' if len(joint_state['positions']) > 5 else ''}")
        print(f"    Velocities:  {[f'{v:.3f}' for v in joint_state['velocities'][:5]]}{'...' if len(joint_state['velocities']) > 5 else ''}")
    else:
        print("  ⚠ No joint state received yet")
    print()
    
    # Test: Get categorized joint state
    print("-" * 70)
    print("[6.5] Testing get_joint_state(categorized=True)")
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
    
    # Test: Get end-effector poses
    print("-" * 70)
    print("[7] Testing get_end_effector_pose()")
    print("-" * 70)
    
    # Left/main arm pose
    left_pose = interface.get_end_effector_pose()
    if left_pose:
        print(f"  ✓ End-effector pose received")
        print(f"    Position:    ({left_pose.position.x:7.3f}, {left_pose.position.y:7.3f}, {left_pose.position.z:7.3f})")
        print(f"    Orientation:  ({left_pose.orientation.x:6.3f}, {left_pose.orientation.y:6.3f}, "
              f"{left_pose.orientation.z:6.3f}, {left_pose.orientation.w:6.3f})")
    else:
        print("  ⚠ No end-effector pose received yet")
    
    # Right arm pose (only for dual-arm mode)
    right_pose = interface.get_right_end_effector_pose()
    if right_pose:
        print(f"\n  ✓ Right arm pose received")
        print(f"    Position:    ({right_pose.position.x:7.3f}, {right_pose.position.y:7.3f}, {right_pose.position.z:7.3f})")
        print(f"    Orientation:  ({right_pose.orientation.x:6.3f}, {right_pose.orientation.y:6.3f}, "
              f"{right_pose.orientation.z:6.3f}, {right_pose.orientation.w:6.3f})")
    elif interface.config.right_end_effector_pose_topic:
        print("\n  ⚠ No right arm pose received yet (dual-arm mode detected)")
    print()
    
    # FSM state cycle configuration
    print("-" * 70)
    print("[8] FSM State Cycle Configuration")
    print("-" * 70)
    # Define FSM state cycle: HOLD → HOME → REST → HOLD → OCS2 → HOLD → MOVEJ (then loop back to HOLD)
    # FSM command values: 0=REST, 1=HOME, 2=HOLD, 3=OCS2/MOVE, 4=MOVEJ
    fsm_states = [2, 1, 100, 2, 3, 2, 4]  # HOLD → HOME → REST → HOLD → OCS2 → HOLD → MOVEJ → (loop back to HOLD)
    fsm_state_names = {100: "REST", 1: "HOME", 2: "HOLD", 3: "OCS2/MOVE", 4: "MOVEJ"}
    # Start from index 0 (HOLD state) since we just switched to it in step [5]
    current_fsm_index = 0  # Index 0 corresponds to HOLD (state 2)
    print(f"  → FSM states cycle: {' → '.join([f'{s}({fsm_state_names[s]})' for s in fsm_states])} → (loop)")
    print(f"  → Switching every 5 seconds")
    print(f"  → Current state: {fsm_states[current_fsm_index]} ({fsm_state_names[fsm_states[current_fsm_index]]})")
    print(f"  → Next switch will go to: {fsm_states[(current_fsm_index + 1) % len(fsm_states)]} ({fsm_state_names[fsm_states[(current_fsm_index + 1) % len(fsm_states)]]})\n")
    
    # Keep running continuously to monitor data updates and cycle FSM states
    print("=" * 70)
    print("[9] Monitoring data continuously and cycling FSM states")
    print("=" * 70)
    is_dual_arm = interface.config.right_end_effector_pose_topic is not None
    if is_dual_arm:
        print("  Mode: DUAL-ARM")
        print("  → In OCS2 state: Arms will extend and retract sequentially")
        print("  → In MOVEJ state: Head and body will move")
    else:
        print("  Mode: SINGLE-ARM")
    print("  → FSM state switches every 5 seconds")
    print("  → Press Ctrl+C to stop and disconnect")
    print("-" * 70 + "\n")
    
    # Arm movement control variables (for OCS2 state)
    # States: idle, left_extending, left_extended, left_retracting, right_extending, right_extended, right_retracting,
    #         head_moving_up, head_moving_down, body_moving_to_target, body_moving_back, completed
    arm_movement_state = "idle"
    arm_movement_start_time = None
    left_initial_pose = None
    right_initial_pose = None
    extend_distance = 0.15  # meters forward
    movement_duration = 2.0  # seconds for each movement phase
    last_fsm_switch_time = 0  # Track when we last switched FSM state
    
    # Head and body control variables (for MOVEJ state)
    head_body_movement_state = "idle"  # idle, head_moving_up, head_moving_down, body_moving_to_target, body_moving_back, completed
    head_body_movement_start_time = None
    head_initial_positions = None
    body_initial_positions = None
    body_target_positions = [-0.899, -1.714, -0.865, 0.0]  # Target body joint positions
    
    try:
        i = 0
        current_fsm_state = fsm_states[current_fsm_index]  # Track current FSM state
        while True:
            time.sleep(1.0)
            i += 1
            
            # Get current states
            joint_state = interface.get_joint_state()
            left_pose = interface.get_end_effector_pose()
            right_pose = interface.get_right_end_effector_pose()
            
            # Control arm movement in OCS2 state (only for dual-arm mode)
            if current_fsm_state == 3 and is_dual_arm:  # OCS2 state
                current_time = time.time()
                
                # Initialize: save initial poses
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
                    
                    arm_movement_state = "left_extending"
                    arm_movement_start_time = current_time
                    # Send left arm extend command
                    left_target = Pose()
                    left_target.position.x = left_initial_pose.position.x + extend_distance
                    left_target.position.y = left_initial_pose.position.y
                    left_target.position.z = left_initial_pose.position.z
                    left_target.orientation = left_initial_pose.orientation
                    interface.send_end_effector_target(left_target)
                    print(f"  [OCS2] → Left arm extending forward...")
                
                # State machine for arm movements
                elif arm_movement_state == "left_extending":
                    if current_time - arm_movement_start_time >= movement_duration:
                        arm_movement_state = "left_extended"
                        arm_movement_start_time = current_time
                        print(f"  [OCS2] → Left arm extended, waiting...")
                
                elif arm_movement_state == "left_extended":
                    if current_time - arm_movement_start_time >= 1.0:  # Wait 1 second
                        arm_movement_state = "left_retracting"
                        arm_movement_start_time = current_time
                        # Send left arm retract command
                        if left_initial_pose:
                            interface.send_end_effector_target(left_initial_pose)
                            print(f"  [OCS2] → Left arm retracting...")
                
                elif arm_movement_state == "left_retracting":
                    if current_time - arm_movement_start_time >= movement_duration:
                        arm_movement_state = "right_extending"
                        arm_movement_start_time = current_time
                        # Send right arm extend command
                        if right_initial_pose:
                            right_target = Pose()
                            right_target.position.x = right_initial_pose.position.x + extend_distance
                            right_target.position.y = right_initial_pose.position.y
                            right_target.position.z = right_initial_pose.position.z
                            right_target.orientation = right_initial_pose.orientation
                            interface.send_right_end_effector_target(right_target)
                            print(f"  [OCS2] → Right arm extending forward...")
                
                elif arm_movement_state == "right_extending":
                    if current_time - arm_movement_start_time >= movement_duration:
                        arm_movement_state = "right_extended"
                        arm_movement_start_time = current_time
                        print(f"  [OCS2] → Right arm extended, waiting...")
                
                elif arm_movement_state == "right_extended":
                    if current_time - arm_movement_start_time >= 1.0:  # Wait 1 second
                        arm_movement_state = "right_retracting"
                        arm_movement_start_time = current_time
                        # Send right arm retract command
                        if right_initial_pose:
                            interface.send_right_end_effector_target(right_initial_pose)
                            print(f"  [OCS2] → Right arm retracting...")
                
                elif arm_movement_state == "right_retracting":
                    if current_time - arm_movement_start_time >= movement_duration:
                        arm_movement_state = "completed"  # Arms completed, ready for state switch
                        arm_movement_start_time = None
                        print(f"  [OCS2] → Both arms completed cycle, ready to switch state...")
            
            # Control head and body joints in MOVEJ state
            elif current_fsm_state == 4:  # MOVEJ state
                current_time = time.time()
                
                # Initialize: save initial positions
                if head_body_movement_state == "idle":
                    categorized_state = interface.get_joint_state(categorized=True)
                    if categorized_state:
                        if categorized_state.get('head', {}).get('positions'):
                            head_initial_positions = categorized_state['head']['positions'].copy()
                        if categorized_state.get('body', {}).get('positions'):
                            body_initial_positions = categorized_state['body']['positions'].copy()
                    
                    head_body_movement_state = "head_moving_up"
                    head_body_movement_start_time = current_time
                    
                    # Start head movement: move up
                    if head_initial_positions and interface.config.head_joint_controller_topic:
                        head_target_up = head_initial_positions.copy()
                        if len(head_target_up) > 0:
                            head_target_up[0] = head_initial_positions[0] + 0.3  # Move up
                        try:
                            interface.send_head_joint_positions(head_target_up)
                            print(f"  [MOVEJ] → Head moving up...")
                        except Exception as e:
                            print(f"  [MOVEJ] ✗ Failed to control head: {e}")
                
                # Head movement: moving up
                elif head_body_movement_state == "head_moving_up":
                    if current_time - head_body_movement_start_time >= movement_duration:
                        head_body_movement_state = "head_moving_down"
                        head_body_movement_start_time = current_time
                        # Move head back down
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            head_target_down = head_initial_positions.copy()
                            if len(head_target_down) > 0:
                                head_target_down[0] = head_initial_positions[0] - 0.3  # Move down
                            try:
                                interface.send_head_joint_positions(head_target_down)
                                print(f"  [MOVEJ] → Head moving down...")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ Failed to control head: {e}")
                
                # Head movement: moving down
                elif head_body_movement_state == "head_moving_down":
                    if current_time - head_body_movement_start_time >= movement_duration:
                        head_body_movement_state = "body_moving_to_target"
                        head_body_movement_start_time = current_time
                        # Return head to initial position
                        if head_initial_positions and interface.config.head_joint_controller_topic:
                            try:
                                interface.send_head_joint_positions(head_initial_positions)
                                print(f"  [MOVEJ] → Head returned to initial position")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ Failed to control head: {e}")
                        
                        # Start body movement: move to target
                        if body_initial_positions and interface.config.body_joint_controller_topic:
                            # Ensure target has same length as current positions
                            body_target = body_target_positions.copy()
                            if len(body_target) < len(body_initial_positions):
                                # Pad with initial positions if target is shorter
                                body_target.extend(body_initial_positions[len(body_target):])
                            elif len(body_target) > len(body_initial_positions):
                                # Truncate if target is longer
                                body_target = body_target[:len(body_initial_positions)]
                            
                            try:
                                interface.send_body_joint_positions(body_target)
                                print(f"  [MOVEJ] → Body moving to target: {[f'{p:.3f}' for p in body_target]}")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ Failed to control body: {e}")
                
                # Body movement: moving to target
                elif head_body_movement_state == "body_moving_to_target":
                    if current_time - head_body_movement_start_time >= movement_duration:
                        head_body_movement_state = "body_moving_back"
                        head_body_movement_start_time = current_time
                        # Return body to initial position
                        if body_initial_positions and interface.config.body_joint_controller_topic:
                            try:
                                interface.send_body_joint_positions(body_initial_positions)
                                print(f"  [MOVEJ] → Body returning to initial position: {[f'{p:.3f}' for p in body_initial_positions]}")
                            except Exception as e:
                                print(f"  [MOVEJ] ✗ Failed to control body: {e}")
                
                # Body movement: moving back
                elif head_body_movement_state == "body_moving_back":
                    if current_time - head_body_movement_start_time >= movement_duration:
                        head_body_movement_state = "completed"  # Mark as completed, ready for state switch
                        head_body_movement_start_time = None
                        head_initial_positions = None
                        body_initial_positions = None
                        print(f"  [MOVEJ] → All movements completed (head, body), ready to switch state...")
            
            # Reset head/body movement when leaving MOVEJ state
            elif current_fsm_state != 4 and head_body_movement_state not in ["idle", "completed"]:
                head_body_movement_state = "idle"
                head_body_movement_start_time = None
                head_initial_positions = None
                body_initial_positions = None
            elif current_fsm_state != 4 and head_body_movement_state == "completed":
                # Reset after leaving MOVEJ state
                head_body_movement_state = "idle"
                head_body_movement_start_time = None
                head_initial_positions = None
                body_initial_positions = None
            
            # Reset arm movement when leaving OCS2 state
            elif current_fsm_state != 3 and arm_movement_state not in ["idle", "completed"]:
                arm_movement_state = "idle"
                arm_movement_start_time = None
                left_initial_pose = None
                right_initial_pose = None
                head_initial_positions = None
                body_initial_positions = None
            elif current_fsm_state != 3 and arm_movement_state == "completed":
                # Reset after leaving OCS2 state
                arm_movement_state = "idle"
                arm_movement_start_time = None
                left_initial_pose = None
                right_initial_pose = None
                head_initial_positions = None
                body_initial_positions = None
            
            # Display status every second
            status_line = f"[{i:3d}s] "
            
            # Display joint state (simplified)
            if joint_state and len(joint_state['positions']) > 0:
                first_joint_pos = joint_state['positions'][0] if joint_state['positions'] else 0.0
                status_line += f"Joints: {len(joint_state['names'])} | "
            
            # Display end-effector poses
            if left_pose:
                status_line += f"Left: ({left_pose.position.x:6.3f}, {left_pose.position.y:6.3f}, {left_pose.position.z:6.3f})"
            if right_pose:
                status_line += f" | Right: ({right_pose.position.x:6.3f}, {right_pose.position.y:6.3f}, {right_pose.position.z:6.3f})"
            
            print(status_line)
            
            # Display categorized joint states every 5 seconds
            if i % 5 == 0:
                categorized_joint_state = interface.get_joint_state(categorized=True)
                if categorized_joint_state:
                    is_dual_arm = interface.config.right_end_effector_pose_topic is not None
                    print()
                    print(f"  [{i}s] Categorized Joint States:")
                    print("-" * 70)
                    
                    if is_dual_arm:
                        # Dual-arm mode
                        if categorized_joint_state.get('left_arm', {}).get('names'):
                            left_arm = categorized_joint_state['left_arm']
                            print(f"  Left Arm ({len(left_arm['names'])} joints):")
                            for j, name in enumerate(left_arm['names']):
                                pos = left_arm['positions'][j] if j < len(left_arm['positions']) else 0.0
                                vel = left_arm['velocities'][j] if j < len(left_arm['velocities']) else 0.0
                                print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                        
                        if categorized_joint_state.get('right_arm', {}).get('names'):
                            right_arm = categorized_joint_state['right_arm']
                            print(f"  Right Arm ({len(right_arm['names'])} joints):")
                            for j, name in enumerate(right_arm['names']):
                                pos = right_arm['positions'][j] if j < len(right_arm['positions']) else 0.0
                                vel = right_arm['velocities'][j] if j < len(right_arm['velocities']) else 0.0
                                print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    else:
                        # Single-arm mode
                        if categorized_joint_state.get('arm', {}).get('names'):
                            arm = categorized_joint_state['arm']
                            print(f"  Arm ({len(arm['names'])} joints):")
                            for j, name in enumerate(arm['names']):
                                pos = arm['positions'][j] if j < len(arm['positions']) else 0.0
                                vel = arm['velocities'][j] if j < len(arm['velocities']) else 0.0
                                print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
                    if categorized_joint_state.get('head', {}).get('names'):
                        head = categorized_joint_state['head']
                        print(f"  Head ({len(head['names'])} joints):")
                        for j, name in enumerate(head['names']):
                            pos = head['positions'][j] if j < len(head['positions']) else 0.0
                            vel = head['velocities'][j] if j < len(head['velocities']) else 0.0
                            print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
                    if categorized_joint_state.get('body', {}).get('names'):
                        body = categorized_joint_state['body']
                        print(f"  Body ({len(body['names'])} joints):")
                        for j, name in enumerate(body['names']):
                            pos = body['positions'][j] if j < len(body['positions']) else 0.0
                            vel = body['velocities'][j] if j < len(body['velocities']) else 0.0
                            print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
                    if categorized_joint_state.get('other', {}).get('names'):
                        other = categorized_joint_state['other']
                        print(f"  Other ({len(other['names'])} joints):")
                        for j, name in enumerate(other['names']):
                            pos = other['positions'][j] if j < len(other['positions']) else 0.0
                            vel = other['velocities'][j] if j < len(other['velocities']) else 0.0
                            print(f"    {name:25s} | Position: {pos:8.3f} | Velocity: {vel:8.3f}")
                    
                    print("-" * 70)
                    print()
            
            # Check if we should switch FSM state
            # Switch every 5 seconds, but wait for OCS2/MOVEJ actions to complete
            # Also switch immediately when actions complete
            should_switch_state = False
            switch_reason = ""
            
            # Check if it's time to switch (every 5 seconds)
            if i % 5 == 0 and i > 0 and i != last_fsm_switch_time:
                # Check if we're in OCS2 state and actions are not completed
                if current_fsm_state == 3 and is_dual_arm and arm_movement_state != "completed":
                    # Delay state switch until actions complete
                    print(f"  [8] ⏳ Waiting for arm movements to complete before switching state...")
                    print(f"      Current movement state: {arm_movement_state}")
                # Check if we're in MOVEJ state and actions are not completed
                elif current_fsm_state == 4 and head_body_movement_state != "completed":
                    # Delay state switch until actions complete
                    print(f"  [8] ⏳ Waiting for head/body movements to complete before switching state...")
                    print(f"      Current movement state: {head_body_movement_state}")
                else:
                    should_switch_state = True
                    switch_reason = "5-second interval"
            
            # Also switch immediately when OCS2 actions complete (even if not at 5-second mark)
            if current_fsm_state == 3 and is_dual_arm and arm_movement_state == "completed" and i != last_fsm_switch_time:
                should_switch_state = True
                switch_reason = "actions completed"
            
            # Also switch immediately when MOVEJ actions complete (even if not at 5-second mark)
            if current_fsm_state == 4 and head_body_movement_state == "completed" and i != last_fsm_switch_time:
                should_switch_state = True
                switch_reason = "actions completed"
            
            # Perform state switch if conditions are met
            if should_switch_state:
                print()
                if switch_reason == "actions completed":
                    print(f"  [8] → Actions completed, switching state immediately...")
                # Move to next FSM state
                current_fsm_index = (current_fsm_index + 1) % len(fsm_states)
                next_fsm_state = fsm_states[current_fsm_index]
                next_fsm_name = fsm_state_names[next_fsm_state]
                
                # Check if FSM state changed, reset movements if leaving OCS2 or MOVEJ
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
                
                try:
                    interface.send_fsm_command(next_fsm_state)
                    print(f"  [8] ✓ FSM state switched to: {next_fsm_state} ({next_fsm_name})")
                    # Update tracked FSM state
                    current_fsm_state = next_fsm_state
                    last_fsm_switch_time = i  # Record when we switched
                except Exception as e:
                    print(f"  [8] ✗ Failed to switch FSM state to {next_fsm_state}: {e}")
                print()
                
    except KeyboardInterrupt:
        print("\n" + "-" * 70)
        print("  Interrupted by user")
        print("-" * 70)
    
    # Disconnect
    print("\n" + "=" * 70)
    print("[10] Disconnecting...")
    interface.disconnect()
    print("  ✓ Interface disconnected successfully!")
    print("=" * 70)
    print("\n  Test completed!\n")
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
