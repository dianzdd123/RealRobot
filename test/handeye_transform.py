import pyrealsense2 as rs
import numpy as np
import cv2
import time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R

# 官方手眼标定参数类
class HandEyeCalibration:
    def __init__(self):
        self.translation = np.array([
            0.06637719970480799,
            -0.032133912794949385,
            0.02259679892714925
        ])
        self.rotation_quaternion = np.array([
            0.0013075170827278532,
            -0.0024892917521336377,
            0.7106597907015502,
            0.7035302095188808
        ])
        self.camera_to_tcp_transform = self._build_transform_matrix()
        print("[INFO] 官方手眼标定参数加载完成")
        print(f"[INFO] 平移向量: {self.translation}")
        print(f"[INFO] 旋转四元数: {self.rotation_quaternion}")
    
    def _build_transform_matrix(self):
        rotation = R.from_quat(self.rotation_quaternion)
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = self.translation
        return transform
    
    def camera_to_world(self, camera_point, tcp_pose, verbose=True):
        try:
            if verbose:
                print(f"[DEBUG] 输入相机坐标: {camera_point} (米)")
                print(f"[DEBUG] 当前TCP位姿: {tcp_pose}")
            
            camera_point_homo = np.append(camera_point, 1.0)
            tcp_point_homo = self.camera_to_tcp_transform @ camera_point_homo
            tcp_point = tcp_point_homo[:3]

            grasp_offset_tcp = np.array([0.05, 0.0, 0.0])
            tcp_point += grasp_offset_tcp

            if verbose:
                print(f"[DEBUG] TCP坐标系中的点（含补偿）: {tcp_point} (米)")
            
            tcp_x, tcp_y, tcp_z = tcp_pose[:3]
            tcp_roll, tcp_pitch, tcp_yaw = tcp_pose[3:]

            tcp_rotation = R.from_euler('xyz', [tcp_roll, tcp_pitch, tcp_yaw], degrees=True)
            tcp_to_world_matrix = np.eye(4)
            tcp_to_world_matrix[:3, :3] = tcp_rotation.as_matrix()
            tcp_to_world_matrix[:3, 3] = [tcp_x / 1000.0, tcp_y / 1000.0, tcp_z / 1000.0]

            tcp_point_homo_world = np.append(tcp_point, 1.0)
            world_point_homo = tcp_to_world_matrix @ tcp_point_homo_world
            world_point = world_point_homo[:3] * 1000  # 转换为 mm

            if verbose:
                print(f"[DEBUG] 世界坐标: {world_point} (mm)")
            
            return world_point
        except Exception as e:
            print(f"[ERROR] 坐标转换失败: {e}")
            return None

# 创建标定对象
hand_eye_calib = HandEyeCalibration()

# 获取相机坐标系中的3D点
def get_3d_camera_coords(pixel_x, pixel_y, depth_frame, intrinsics):
    depth = depth_frame.get_distance(pixel_x, pixel_y)
    if depth == 0:
        print(f"[WARN] 深度为 0，({pixel_x}, {pixel_y}) 无效")
        return None
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth)
    return np.array(point_3d, dtype=np.float32)

# 鼠标点击处理函数
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color_image, depth_frame, intrinsics = param
        print(f"[INFO] 点击像素点: ({x}, {y})")
        cam_point = get_3d_camera_coords(x, y, depth_frame, intrinsics)
        if cam_point is not None:
            print(f"[INFO] 相机坐标: {np.round(cam_point, 4)} (米)")
            tcp_pose = arm.get_position(is_radian=False)[1]
            world_point = hand_eye_calib.camera_to_world(cam_point, tcp_pose)
            print(f"[RESULT] 世界坐标: {np.round(world_point, 2)} (mm)")
            if world_point is None:
                print("[ERROR] 坐标转换失败，跳过")
                return

            # 限制 Z 值不低于 300mm
            if world_point[2] < 300:
                print(f"[WARN] Z 值过低（{world_point[2]:.1f}mm），自动设置为 300mm")
                world_point[2] = 300

            print(f"[RESULT] 世界坐标（用于移动）: {np.round(world_point, 2)} (mm)")

            # 执行机械臂移动
            ret = arm.set_position(
                x=world_point[0],
                y=world_point[1],
                z=world_point[2],
                roll=180,
                pitch=0,
                yaw=0,
                speed=50,
                wait=True
            )
            if ret == 0:
                print("[INFO] 机械臂移动完成 ✅")
            else:
                print(f"[ERROR] 机械臂移动失败，错误码: {ret}")

# 主循环：图像显示 + 鼠标点击监听
try:
    # 初始化 RealSense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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

    arm.set_position(x=-112, y=-400, z=460, roll=180, pitch=0, yaw=45, speed=75, wait=True)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        cv2.imshow("Click to test world position", color_image)
        cv2.setMouseCallback("Click to test world position", on_mouse_click, param=(color_image, depth_frame, intrinsics))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    pipeline.stop()
