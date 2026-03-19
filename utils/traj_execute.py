"""
Trajectory execution module for translating planned JSON steps into xArm physical motions.
"""
import json
import time

def move_arm_to_target(arm, x, y, z, yaw=0, gripper_value=850, speed=80):
    """Executes a coordinated point-to-point motion with gripper actuation."""
    # Standard orientation: roll=180, pitch=0
    arm.set_position(x=x, y=y, z=z, roll=180, pitch=0, yaw=yaw, speed=speed, wait=True)
    arm.set_gripper_position(gripper_value, wait=True)

def execute(json_path, arm=None):
    """Parses a trajectory JSON file and executes sequence of robotic commands."""
    if arm is None:
        print("[ERROR] No xArm instance provided to execute().")
        return

    with open(json_path, 'r') as f:
        steps = json.load(f)

    print(f"[INFO] Starting execution for {len(steps)} planned steps.")

    for i, step in enumerate(steps):
        pos = step.get('position')
        yaw = step.get('yaw', 0)
        gripper_value = step.get('gripper', 850)

        # Validate coordinate data
        if not pos or len(pos) != 3:
            print(f"[WARN] Step {i+1} has invalid position data, skipping.")
            continue

        # Apply safety Z-clipping at 185mm to prevent table collision
        target_z = max(pos[2], 185)
        
        print(f"[INFO] Step {i+1}/{len(steps)}: Moving to [{pos[0]:.1f}, {pos[1]:.1f}, {target_z:.1f}] | Yaw: {yaw} | Gripper: {gripper_value}")
        
        move_arm_to_target(
            arm,
            x=pos[0],
            y=pos[1],
            z=target_z,
            yaw=yaw,
            gripper_value=gripper_value
        )
        
        # Short stabilization pause
        time.sleep(1)

    print("[INFO] Trajectory execution completed successfully.")