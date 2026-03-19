import numpy as np
import pyrealsense2 as rs
from .occluded import robust_center_depth

def get_3d_coords_from_centers(npz_path, pixel_points, depth_intrin):
    """Deprojects 2D pixel coordinates to 3D camera coordinates using RGB-D data and intrinsics."""
    data = np.load(npz_path)
    depth_image = data['depth']  # shape: (H, W), dtype=uint16, unit: mm
    H, W = depth_image.shape

    # 1) Extract points list, compatible with dict {"points": [...]} or direct list formats
    if pixel_points is None:
        pts = []
    elif isinstance(pixel_points, dict) and 'points' in pixel_points:
        pts = pixel_points.get('points', [])
    else:
        pts = pixel_points

    # 2) Sanitize: Ensure point length is 2 and handle None placeholders
    clean_pts = []
    for p in pts:
        if p is None or not isinstance(p, (list, tuple)) or len(p) != 2:
            clean_pts.append(None)
        else:
            clean_pts.append((p[0], p[1]))

    # 3) Convert normalized [0,1] coordinates to pixel coordinates
    def to_uv(p):
        if p is None:
            return None
        x, y = p
        if x is None or y is None:
            return None
        
        # Handle normalized coordinates [0, 1]
        if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
            u = int(round(float(x) * (W - 1)))
            v = int(round(float(y) * (H - 1)))
        else:
            u = int(round(float(x)))
            v = int(round(float(y)))
        return (u, v)

    uv_pts = [to_uv(p) for p in clean_pts]

    # 4) Retrieve depth and deproject to 3D camera coordinates
    coords_3d = []
    for uv in uv_pts:
        if uv is None:
            coords_3d.append(None)
            continue

        u, v = uv
        if not (0 <= u < W and 0 <= v < H):
            coords_3d.append(None)
            print(f"[WARN] Out of image bounds: ({u}, {v})")
            continue

        depth_mm = robust_center_depth(depth_image, u, v, win=5)
        if depth_mm is None or depth_mm <= 0:
            coords_3d.append(None)
            print(f"[WARN] Invalid depth: ({u}, {v}) → {depth_mm}")
            continue

        depth_m = depth_mm / 1000.0  # Convert mm to meters
        coord_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_m)
        coords_3d.append(coord_3d)

    return coords_3d