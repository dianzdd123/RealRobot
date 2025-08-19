from xarm.wrapper import XArmAPI

'''
控制机械臂到指定位置，目的是快速移动机械臂，测试世界坐标是否准确
'''
arm = XArmAPI('192.168.1.117')  # Connect to robot

arm.motion_enable(enable=True)
arm.set_state(0)     # Set to ready state
arm.set_mode(0)      # Set to position control mode

# Move robot to initial viewing position
# arm.set_position(x=221.7, y=-3.4, z=450, roll=180, pitch=0, yaw=0, speed=65, wait=True)
# arm.set_position(x=0, y=-450, z=550, roll=180, pitch=0, yaw=0, speed=65, wait=True)
arm.set_position(x=123, y=-575.59, z=350, roll=180, pitch=0, yaw=0, speed=50, wait=True)
arm.set_gripper_position(850, wait=True)
