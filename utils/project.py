import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

# ---------- 相机 → 世界坐标变换辅助函数 ----------
def build_transform(position, orientation_deg):
    """ 构建齐次变换矩阵 T (4x4) """
    rot = R.from_euler('xyz', orientation_deg, degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = position
    return T

def camera_to_world(camera_point, tool_pos_mm, tool_orient_deg, cam_offset, cam_orient):
    """ 从相机坐标系 → 世界坐标系 """
    tool_pos_m = [x / 1000.0 for x in tool_pos_mm]
    T_base_to_tool = build_transform(tool_pos_m, tool_orient_deg)
    T_tool_to_camera = build_transform(cam_offset, cam_orient)
    T_base_to_camera = T_base_to_tool @ T_tool_to_camera
    P_cam = np.array([*camera_point, 1.0])
    P_world = T_base_to_camera @ P_cam
    return (P_world[:3] * 1000).tolist()  # 以 mm 返回
    # return P_world[:3] 

arm = XArmAPI('192.168.1.117')  # Connect to robot

arm.motion_enable(enable=True)
arm.set_state(0)     # Set to ready state
arm.set_mode(0)      # Set to position control mode

# Move robot to initial viewing position
# arm.set_position(x=220, y=0, z=450, roll=180, pitch=0, yaw=0, speed=65, wait=True)
arm.set_position(x=0, y=-450, z=500, roll=180, pitch=0, yaw=0, speed=65, wait=True)
arm.set_gripper_position(850, wait=True)

# ---------- 设置相机安装参数 ----------
# cam_offset = [0.13352, 0, 0.0]   # 相机在末端 x 正方向 75mm（单位米）
# cam_orient = [0, 0, 90]           # x↔y，z 轴相反（已分析）

cam_offset = [0.066, -0.032, 0.023]  # 单位：米
cam_orient = [0.162, -0.270, 89.836]  # 由标定四元数转换得到的姿态角度（度）

# ---------- 初始化 RealSense ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 等待相机稳定
for _ in range(30):
    pipeline.wait_for_frames()

# 获取一帧
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# 获取内参
depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

# 转换图像为 numpy 格式
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# ---------- 点击图像获取像素点 ----------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        u, v = x, y
        depth = depth_frame.get_distance(u, v)
        if depth == 0:
            print(f"[WARN] 像素点 ({u},{v}) 无深度值")
            return
        print(f"[INFO] Depth at ({u},{v}): {depth:.3f} m")

        # 相机坐标系 3D 点
        camera_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth)
        print("[INFO] Camera Point:", ["{:.2f}".format(v) for v in camera_point[:3]])

        code, pos = arm.get_position(is_radian=False)
        tool_pos_mm = pos[0:3]
        tool_orient_deg = pos[3:6]

        # 转换为世界坐标
        world_point = camera_to_world(camera_point, tool_pos_mm, tool_orient_deg, cam_offset, cam_orient)
        print(f"World Point: x={world_point[0]:.2f} mm, y={world_point[1]:.2f} mm, z={world_point[2]:.2f} mm\n")


# ---------- 显示图像并注册点击事件 ----------
cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Color Image", mouse_callback)

while True:
    cv2.imshow("Color Image", color_image)
    if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 退出
        break

pipeline.stop()
cv2.destroyAllWindows()
