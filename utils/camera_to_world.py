import numpy as np
from scipy.spatial.transform import Rotation as R

# Official calibration parameters
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

_offset = np.array([0.035-0.0264, 0.0, -0.132])  # TCP grasping point compensation (meters)

def camera_to_world(camera_point, tcp_pose, min_z_mm=220, verbose=False):
    """Transforms camera coordinates to robot world coordinates in millimeters."""
    try:
        # Camera to TCP transformation
        rotation = R.from_quat(_rotation_quaternion)
        T_cam2tcp = np.eye(4)
        T_cam2tcp[:3, :3] = rotation.as_matrix()
        T_cam2tcp[:3, 3] = _translation

        cam_point_h = np.append(camera_point, 1.0)
        tcp_point = T_cam2tcp @ cam_point_h
        tcp_point[:3] += _offset

        # TCP to World transformation
        tcp_pos = np.array(tcp_pose[:3]) / 1000.0  # mm to m
        tcp_rot = R.from_euler('xyz', tcp_pose[3:], degrees=True)
        T_tcp2world = np.eye(4)
        T_tcp2world[:3, :3] = tcp_rot.as_matrix()
        T_tcp2world[:3, 3] = tcp_pos

        world_point = T_tcp2world @ np.append(tcp_point[:3], 1.0)
        world_point_mm = world_point[:3] * 1000  # m to mm

        # Apply safety Z-clipping
        if world_point_mm[2] < min_z_mm:
            if verbose:
                print(f"[WARN] Z value is lower than {min_z_mm}mm, clipped to safety limit.")
            world_point_mm[2] = min_z_mm

        if verbose:
            print(f"[DEBUG] World coordinates: {np.round(world_point_mm, 2)} mm")

        return world_point_mm

    except Exception as e:
        print(f"[ERROR] camera_to_world transformation failed: {e}")
        return None