import pyrealsense2 as rs
import numpy as np
import cv2
import time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度标定信息
align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth Scale: {depth_scale}")

# 初始化机械臂
arm = XArmAPI('192.168.1.117')
time.sleep(0.5)
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

# 清除机械臂的报警和错误
if arm.warn_code != 0:
    arm.clean_warn()
if arm.error_code != 0:
    arm.clean_error()

def camera_to_world(camera_point, cam_position, cam_orientation, verbose=True):
    # Step 1: 坐标轴重排
    if np.allclose(camera_point, [0, 0, 0]):
        camera_point = np.array([0, 0, -100.0], dtype=np.float32)
    x_c, y_c, z_c = camera_point
    camera_point_reordered = np.array([(y_c+65), (x_c-30), -(z_c-100)])  # 相机→世界方向转换

    if verbose:
        print(f"[DEBUG] 原始相机坐标:       {camera_point}")
        print(f"[DEBUG] 坐标轴重排后坐标:   {camera_point_reordered}")

    # Step 2: 生成旋转矩阵
    roll, pitch, yaw = cam_orientation
    rotation = R.from_euler('XYZ', [roll, pitch, yaw], degrees=True)
    R_wc = rotation.as_matrix()

    if verbose:
        print(f"[DEBUG] 旋转矩阵 R_wc:\n{R_wc}")

    # Step 3: 平移
    T_wc = np.array(cam_position)


    # Step 4: 应用旋转 + 平移
    world_point = R_wc @ camera_point_reordered + T_wc

    if verbose:
        print(f"[DEBUG] 世界坐标下的点:     {world_point}")

    return world_point, camera_point_reordered, R_wc

# 鼠标点击回调函数
def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return
    
    print(f"{x},{y}")
    depth_value = depth_frame.get_distance(x, y)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    cam_x, cam_y, cam_z = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_value)
    print(f"[INFO] 目标点坐标: x={cam_x:.3f}, y={cam_y:.3f}, z={cam_z:.3f}")

    # 获取当前机械臂位置
    position = arm.get_position(is_radian=False)[1]
    print(f"[INFO] 当前机械臂位置: {position}")
    
    curr_x, curr_y, curr_z,curr_ro,curr_pi,curr_ya = position[0], position[1], position[2],position[3], position[4], position[5]
    camera_point = [cam_x*1000,cam_y*1000,cam_z*1000]
    cam_position = [curr_x,curr_y,curr_z]
    cam_orientation = [abs(curr_ro)-180,curr_pi,curr_ya]
    world_point, reordered_point, R_wc = camera_to_world(camera_point, cam_position, cam_orientation)
    print(f"[INFO]位置坐标: {world_point}")
    arm_x = world_point[0]
    arm_y = world_point[1]
    arm_z = max(world_point[2],200)  # 保持 z 方向加一个固定偏移

    # 检查是否在机械臂工作范围内
    if 120 <= arm_x <= 500 and -200 <= arm_y <= 200:
        arm.set_position(x=arm_x, y=arm_y, z=arm_z, roll=179, pitch=0, yaw=0, speed=50, wait=True)
        print("[INFO] 已发送目标位置")
    else:
        print("[WARN] 目标点超出工作空间范围！")

# 创建窗口并绑定回调
cv2.namedWindow('RealSense')
cv2.setMouseCallback('RealSense', mouse_callback)

# 主循环
try:
    # 初始位置
    arm.set_position(x=250.7, y=-3.4, z=350, roll=179, pitch=0, yaw=0, speed=50, wait=True)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)

        # 按下 ESC 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
    arm.disconnect()
    print("[INFO] 已退出程序")
