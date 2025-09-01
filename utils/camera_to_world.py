import numpy as np
from scipy.spatial.transform import Rotation as R

# 官方标定参数
_translation = np.array([
    0.06637719970480799,
    -0.032133912794949385,
    0.02259679892714925
])

_rotation_quaternion = np.array([
    0.0013075170827278532,
    -0.0024892917521336377,
    0.7106597907015502,
    0.7035302095188808
])

_offset = np.array([0.035-0.0264, 0.0, -0.132])  # TCP 抓取点补偿（米）

# 相机坐标 → 世界坐标
def camera_to_world(camera_point, tcp_pose, min_z_mm=220, verbose=False):
    """
    将相机坐标转换为世界坐标（单位：mm）

    Args:
        camera_point: 相机坐标 [x, y, z] (米)
        tcp_pose: TCP 当前末端位姿 [x, y, z, roll, pitch, yaw] (mm, deg)
        min_z_mm: 最小 z 值（毫米）
        verbose: 是否打印调试信息

    Returns:
        world_point_mm: 世界坐标 [x, y, z] (毫米)
    """
    try:
        # 相机 → TCP
        rotation = R.from_quat(_rotation_quaternion)
        T_cam2tcp = np.eye(4)
        T_cam2tcp[:3, :3] = rotation.as_matrix()
        T_cam2tcp[:3, 3] = _translation

        cam_point_h = np.append(camera_point, 1.0)
        tcp_point = T_cam2tcp @ cam_point_h
        tcp_point[:3] += _offset

        # TCP → World
        tcp_pos = np.array(tcp_pose[:3]) / 1000.0  # mm → m
        tcp_rot = R.from_euler('xyz', tcp_pose[3:], degrees=True)
        T_tcp2world = np.eye(4)
        T_tcp2world[:3, :3] = tcp_rot.as_matrix()
        T_tcp2world[:3, 3] = tcp_pos

        world_point = T_tcp2world @ np.append(tcp_point[:3], 1.0)
        world_point_mm = world_point[:3] * 1000  # m → mm

        if world_point_mm[2] < min_z_mm:
            # if verbose:
            #     print(f"[WARN] Z 值低于 {min_z_mm}，已修正")
            world_point_mm[2] = min_z_mm

        if verbose:
            print(f"[DEBUG] 世界坐标: {np.round(world_point_mm, 2)} mm")

        return world_point_mm

    except Exception as e:
        print(f"[ERROR] camera_to_world 转换失败: {e}")
        return None
