from xarm.wrapper import XArmAPI
import time

def test_get_position(robot_ip='192.168.1.117'):
    # 连接机械臂
    arm = XArmAPI(robot_ip)
    time.sleep(0.5)

    # 启用运动
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    # 获取末端位置
    code, position = arm.get_position(is_radian=False)

    if code != 0:
        print(f"[ERROR] 获取位姿失败，错误码：{code}")
        return

    x, y, z = position[0], position[1], position[2]
    roll, pitch, yaw = position[3], position[4], position[5]

    print(f"末端位置（mm）: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    print(f"末端姿态（deg）: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")

if __name__ == "__main__":
    test_get_position()
