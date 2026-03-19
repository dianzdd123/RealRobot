import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

def compute_gripper_for_real(width_m, gripper_min=0, gripper_max=850, width_min=0.01, width_max=0.08, safety_margin=0.005):
    """Maps object width in meters to real xArm gripper values (0-850)."""
    width_clamped = np.clip(width_m, width_min, width_max)
    gripper_range = gripper_max - gripper_min

    # Linear mapping from width to gripper value
    ratio = (width_clamped - width_min) / (width_max - width_min)
    close_val = gripper_min + ratio * gripper_range

    # Calculate open value with an additional safety margin
    open_width = np.clip(width_clamped + safety_margin, width_min, width_max)
    open_ratio = (open_width - width_min) / (width_max - width_min)
    open_val = gripper_min + open_ratio * gripper_range

    return int(round(open_val)), int(round(close_val))

def compute_gripper_for_sim(w_obj_m, gripper_min=0.0, gripper_max=0.85, safety_margin=0.005, max_obj_width=0.08):
    """Computes gripper open/close values for simulation environments."""
    w = np.clip(w_obj_m, 0.01, max_obj_width)
    gripper_range = gripper_max - gripper_min

    # Mapping: 0.0 width -> fully open, max_obj_width -> fully closed
    ratio = w / max_obj_width
    base_value = gripper_min + ratio * gripper_range
    
    open_value = np.clip(base_value + safety_margin * gripper_range / max_obj_width, gripper_min, gripper_max)
    close_value = np.clip(base_value - safety_margin * gripper_range / max_obj_width, gripper_min, gripper_max)

    return open_value, close_value

def infer_grasp_angle_and_width(mask, draw_on=None, object_index=0, pixel_to_meter=0.0001):
    """Calculates grasp orientation (PCA) and object width along the minor axis."""
    ys, xs = np.nonzero(mask)
    coords = np.stack((xs, ys), axis=1)

    if len(coords) < 10:
        print(f"[WARN] Mask for object {object_index} is too small, skipping.")
        dummy_img = draw_on if draw_on is not None else np.zeros((*mask.shape, 3), dtype=np.uint8)
        return 0, 0, dummy_img, np.array([1, 0]), np.array([0, 1])

    # 1. Centralize coordinates
    mean = np.mean(coords, axis=0)
    centered = coords - mean

    # 2. Perform PCA to find major and minor axes
    # 
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    minor_axis = eigvecs[:, np.argmin(eigvals)]

    # 3. Calculate grasp angle around Z-axis
    angle_rad = np.arctan2(minor_axis[1], minor_axis[0])
    grasp_angle = -np.degrees(angle_rad)
    
    # 4. Project points onto minor axis to calculate width
    projections = centered @ minor_axis
    width_pixels = projections.max() - projections.min()
    width = width_pixels * pixel_to_meter if pixel_to_meter else width_pixels

    # 5. Visualization logic
    output_img = draw_on.copy() if draw_on is not None else cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    center_int = tuple(np.round(mean).astype(int))

    for vec, color, label in zip([major_axis, minor_axis], [(0, 255, 0), (0, 0, 255)], ["major", "minor"]):
        tip = (mean + vec * 50).astype(int)
        cv2.arrowedLine(output_img, center_int, tuple(tip), color=color, thickness=2, tipLength=0.2)
        cv2.putText(output_img, f"{label}_{object_index}", tuple(tip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return grasp_angle, width, output_img, major_axis, minor_axis

def extract_grasp_infos(image_rgb, masks, pixel_to_meter=0.0001, output_path="imgs/grasp_infos.png"):
    """Extracts grasp parameters (center, angle, width) for all detected masks."""
    draw_img = image_rgb.copy()
    grasp_infos = []

    for i, mask in enumerate(masks):
        angle, width, draw_img, _, _, = infer_grasp_angle_and_width(
            mask=mask,
            draw_on=draw_img,
            object_index=i,
            pixel_to_meter=pixel_to_meter
        )
        
        # Calculate mask center using pixel coordinate averaging
        cy, cx = np.mean(np.argwhere(mask), axis=0).astype(int)

        grasp_infos.append({
            "index": i,
            "center": (cx, cy),
            "angle": round(angle, 2),
            "width": round(width, 2),
        })

        # Visualize center point and index
        cv2.circle(draw_img, (cx, cy), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.putText(draw_img, f"#{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # Save final visualization result
    draw_img_bgr = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, draw_img_bgr)

    return grasp_infos, output_path
 
def simple_quaternion_from_angle(angle_deg):
    """Generates a simplified unit quaternion from a 2D rotation angle."""
    # Map angle to sine curve for the y-component
    y = np.sin(np.radians(angle_deg))

    # Construct custom quaternion structure [x, y, z, w]
    quat = np.array([0.0, y, 1.0, 0.0])

    # Normalize to create a unit quaternion
    quat = quat / np.linalg.norm(quat)

    return np.round(quat, 4)