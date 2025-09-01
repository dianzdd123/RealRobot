import numpy as np
import pyrealsense2 as rs
from .occluded import robust_center_depth


# def get_3d_coords_from_centers(npz_path, pixel_points, depth_intrin):
#     """
#     从像素点列表中提取相机坐标系下的 3D 坐标（使用原始 uint16 深度图）

#     参数:
#         npz_path (str): 保存 color/depth 的 .npz 文件，depth 为 uint16 毫米单位
#         pixel_points (list of (u, v)): 要查询的像素坐标列表
#         depth_intrin (rs.intrinsics): RealSense 内参对象

#     返回:
#         coords_3d (list of [x, y, z] or None): 每个像素点对应的相机坐标系下的 3D 坐标（单位米）
#     """
#     data = np.load(npz_path)
#     depth_image = data['depth']  # shape: (H, W), dtype=uint16, 单位 mm

#     coord_list = []

#     for (u, v) in np.array(pixel_points).astype(int):
#         if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
#             depth_mm = depth_image[v, u]
#             if depth_mm > 0:
#                 depth_m = depth_mm / 1000.0  # 转换为米
#                 coord_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
#                 coord_list.append(coord_3d)
#             else:
#                 coord_list.append(None)
#                 print(f"无效深度: ({u}, {v})")
#         else:
#             coord_list.append(None)
#             print(f"超出图像范围: ({u}, {v})")

#     return coord_list

import numpy as np
import pyrealsense2 as rs

def get_3d_coords_from_centers(npz_path, pixel_points, depth_intrin):
    """
    从像素点列表中提取相机坐标系下的 3D 坐标（使用原始 uint16 深度图）

    参数:
        npz_path (str): 保存 color/depth 的 .npz 文件，depth 为 uint16 毫米单位
        pixel_points: 可以是 [{"points":[[x1,y1],[x2,y2],...]}] 的字典或 [[x,y], ...] 的列表。
                      坐标既可为像素坐标，也可为 [0,1] 归一化坐标。
        depth_intrin (rs.intrinsics): RealSense 内参对象

    返回:
        coords_3d (list of [x, y, z] or None): 每个像素点对应的相机坐标系下的 3D 坐标（单位米）
    """
    data = np.load(npz_path)
    depth_image = data['depth']  # shape: (H, W), dtype=uint16, 单位 mm
    H, W = depth_image.shape

    # 1) 取出点列表，兼容 {"points":[...]} 或直接列表
    if pixel_points is None:
        pts = []
    elif isinstance(pixel_points, dict) and 'points' in pixel_points:
        pts = pixel_points.get('points', [])
    else:
        pts = pixel_points

    # 2) 清洗：保证长度为2；允许 None 占位（返回 None）
    clean_pts = []
    for p in pts:
        if p is None or not isinstance(p, (list, tuple)) or len(p) != 2:
            clean_pts.append(None)
        else:
            clean_pts.append((p[0], p[1]))

    # 3) 判断是否为归一化坐标并转换为像素坐标
    def to_uv(p):
        if p is None:
            return None
        x, y = p
        # 允许 x/y 是 None
        if x is None or y is None:
            return None
        # 归一化坐标：[0,1]
        if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
            u = int(round(float(x) * (W - 1)))
            v = int(round(float(y) * (H - 1)))
        else:
            u = int(round(float(x)))
            v = int(round(float(y)))
        return (u, v)

    uv_pts = [to_uv(p) for p in clean_pts]

    # 4) 查深度并反投影
    coords_3d = []
    for uv in uv_pts:
        if uv is None:
            coords_3d.append(None)
            continue

        u, v = uv
        if not (0 <= u < W and 0 <= v < H):
            coords_3d.append(None)
            print(f"超出图像范围: ({u}, {v})")
            continue

        depth_mm = robust_center_depth(depth_image, u, v, win=5)
        if depth_mm <= 0:
            coords_3d.append(None)
            print(f"无效深度: ({u}, {v})")
            continue

        depth_m = depth_mm / 1000.0
        coord_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
        coords_3d.append(coord_3d)

    return coords_3d
