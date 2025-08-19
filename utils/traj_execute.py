import json
import time

def move_arm_to_target(arm, x, y, z, yaw=0, gripper_value=850, speed=80):
    arm.set_position(x=x, y=y, z=z, roll=180, pitch=0, yaw=yaw, speed=speed, wait=True)
    arm.set_gripper_position(gripper_value, wait=True)

def execute(json_path, arm=None):
    with open(json_path, 'r') as f:
        steps = json.load(f)

    for i, step in enumerate(steps):
        pos = step.get('position')
        yaw = step.get('yaw', 0)
        gripper_value = step.get('gripper', 850)

        if not pos or len(pos) != 3:
            print(f"[WARN] Step {i+1} missing or invalid position, skipping.")
            continue

        print(f"[INFO] Executing Step {i+1}: x={pos[0]:.1f}, y={pos[1]:.1f}, z={max(pos[2], 185):.1f}, yaw={yaw}, gripper={gripper_value}")
        
        move_arm_to_target(
            arm,
            x=pos[0],
            y=pos[1],
            z=max(pos[2], 185),
            yaw=yaw,
            gripper_value=gripper_value
        )
        time.sleep(1)
