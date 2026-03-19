"""
Advanced robotic grasping pipeline with occlusion reasoning and affordance 
refinement for cluttered and stacked environments.
"""
# Import utility functions
from utils.yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import get_3d_coords_from_centers
from utils.save_json import save_scene_info_to_json
from utils.rotate import extract_grasp_infos, infer_grasp_angle_and_width
from xarm.wrapper import XArmAPI
from utils.API import subtasks_planner, trajectory_plan_on_real
from utils.camera_to_world import camera_to_world
from utils.traj_execute import execute
from utils.affordance_area import compute_affordance_mask, compute_affordance_striped_fullwidth
from utils.voice_hybrid import wait_for_command

# Import standard libraries
import pyrealsense2 as rs
import sys
import os
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from utils.occluded import check_occlusion, save_occlusion_info, compute_grasp_difficulty
from ultralytics import YOLO

def init_robot_and_camera(robot_ip="192.168.1.117", img_path="color.png", npz_path="frames.npz"):
    """Initializes the xArm robot and RealSense camera to capture aligned RGB-D data."""
    # ========== Step 1: Robot Initialization ==========
    arm = XArmAPI(robot_ip)
    arm.motion_enable(enable=True)
    arm.set_state(0)     # Set to ready state
    arm.set_mode(0)      # Set to position control mode

    # Move robot to initial viewing position
    arm.set_position(x=-70, y=-500, z=500, roll=180, pitch=0, yaw=0, speed=80, wait=True)
    arm.set_gripper_position(850, wait=True)

    # ========== Step 2: RealSense Camera Initialization ==========
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Warm up the camera
    for _ in range(10):
        pipeline.wait_for_frames()

    # Capture one aligned RGB-D frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("[ERROR] Failed to capture frames from RealSense.")

    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Obtain camera intrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    # Save RGB and depth images
    cv2.imwrite(img_path, color_image)
    np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    pipeline.stop()

    return arm, color_image, depth_image, depth_intrin

def compute_initial_affordance(image_rgb, masks, sam_centers):
    """
    Calculates the initial grasp affordance for each detected object.

    Args:
        image_rgb (np.ndarray): The input RGB image.
        masks (list): List of segmentation masks from SAM2.
        sam_centers (list): List of object centers.

    Returns:
        tuple: (affordance_masks, grasp_angles, widths, major_axes, minor_axes)
    """
    affordance_masks, grasp_angles, widths, major_axes, minor_axes = [], [], [], [], []

    for i, m in enumerate(masks):
        angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(
            m, draw_on=image_rgb, object_index=i
        )
        aff_mask = compute_affordance_mask(
            mask=m,
            major_axis=major_axis,
            minor_axis=minor_axis,
            mean=np.array(sam_centers[i], dtype=np.float32),
            center_ratio=0.3,
            expand_pixels=20
        )
        affordance_masks.append(aff_mask)
        grasp_angles.append(angle)
        widths.append(width)
        major_axes.append(major_axis)
        minor_axes.append(minor_axis)

    return affordance_masks, grasp_angles, widths, major_axes, minor_axes

def detect_all_occlusions(depth_image, masks, affordance_masks, sam_centers, class_ids, class_names):
    """Detects occlusions between objects and returns the relationship graph."""
    occlusion_info = {i: [] for i in range(len(masks))}
    for i in range(len(masks)):
        ok, occ_idx, occ_type, _ = check_occlusion(
            depth_image=depth_image,
            mask=masks[i],
            affordance_mask=affordance_masks[i],
            center=sam_centers[i],
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            class_ids=class_ids,
            class_names=class_names,
            self_idx=i
        )
        if ok and occ_idx is not None and occ_idx != i:
            occlusion_info[i].append((occ_idx, occ_type))
    return occlusion_info

def find_mutual_occlusions(occlusion_info):
    """Identifies pairs of objects that mutually occlude each other."""
    mutual_pairs = set()
    for i, rels in occlusion_info.items():
        for j, _t in rels:
            if i in [x for x,_ in occlusion_info.get(j, [])]:
                mutual_pairs.add(tuple(sorted((i, j))))
    return list(mutual_pairs)

def recompute_for_mutual_occlusions(image_rgb, depth_image, masks, sam_centers,
                                    class_ids, class_names, mutual_pairs,
                                    major_axes, minor_axes, affordance_masks,
                                    grasp_angles, widths):
    """
    Handles complex grasping scenarios where two objects mutually occlude each other.
    
    This function implements a refinement strategy for 'deadlock' occlusion cases 
    (e.g., Object A is over Object B, but also partially under it). It utilizes 
    a striped sampling approach to identify valid grasping regions (affordance) 
    that are reachable despite the mutual physical constraints.

    Args:
        image_rgb (np.ndarray): Original RGB image for visualization.
        depth_image (np.ndarray): Depth map used for height-based edge detection.
        masks (list): Original segmentation masks for all objects.
        sam_centers (list): Geometric centers of the objects.
        class_ids (list): Category IDs for detected objects.
        class_names (dict): Mapping from class IDs to human-readable names.
        mutual_pairs (list): List of tuples containing indices of mutually occluding objects.
        major_axes (list): Calculated major axes of object orientations.
        minor_axes (list): Calculated minor axes of object orientations.
        affordance_masks (list): List of graspable region masks to be updated.
        grasp_angles (list): List of grasp orientations (in degrees) to be updated.
        widths (list): List of gripper opening widths to be updated.

    Returns:
        tuple: (updated_affordance_masks, updated_grasp_angles, updated_widths, 
                updated_major_axes, updated_minor_axes)
    """
    MIN_MASK_AREA = 100
    for idx1, idx2 in mutual_pairs:
        for idx in [idx1, idx2]:
            new_mask, _ = compute_affordance_striped_fullwidth(
                idx=idx,
                masks=masks,
                depth_image=depth_image.astype(np.uint16),
                class_ids=class_ids,
                class_names=class_names,
                major_axes=major_axes,
                minor_axes=minor_axes,
                prox_dilate_px=15,
                depth_dilate_px=6,
                depth_margin_mm=20,
                sample_step=2,
                local_win=3,
                stripe_bin_px=1
            )

            # --- 新增错误逻辑 ---
            if np.count_nonzero(new_mask) < MIN_MASK_AREA:
                object_label = class_names.get(class_ids[idx], "Unknown")
                print(f"[ERROR] Object {idx} ({object_label}): No valid graspable region found after recomputation.")
                continue
            # -----------------

            affordance_masks[idx] = new_mask
            angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(
                new_mask, draw_on=image_rgb, object_index=idx
            )
            grasp_angles[idx] = angle
            widths[idx] = width
            major_axes[idx] = major_axis
            minor_axes[idx] = minor_axis

    return affordance_masks, grasp_angles, widths, major_axes, minor_axes

def get_mask_centers(masks):
    """Computes the geometric centers for a list of masks."""
    centers = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            centers.append(None)
        else:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            centers.append([cx, cy])
    return centers

if __name__ == "__main__": 
    # ========== Step 0: Configuration ==========
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"
    
    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names

    # ========== Step 1: Robot & Camera Initialization ==========
    arm, image_rgb, depth_image, depth_intrin = init_robot_and_camera(img_path=img_path, npz_path=npz_path)
    print("[INFO] Robot and camera initialized successfully.")

    # ========== Step 2: Object Detection (YOLO) ==========
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
    if len(boxes) == 0:
        print("[ERROR] No objects detected by YOLO. Exiting.")
        sys.exit(0)

    # ========== Step 2.5: Merging Detections ==========
    mboxes, mclass_ids, mcenters, _ = merge_detections_by_depth_and_proximity(
        boxes=boxes,
        class_ids=class_ids,
        centers_xy=yolo_centers,
        depth_image=depth_image.astype(np.uint16),
        pixel_dist_thresh=100,
        depth_diff_thresh=20,
        same_class_only=True,
        win=3
    )

    # ========== Step 3: Segmentation (SAM2) ==========
    masks, sam_centers, _ = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=mboxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )

    # ========== Step 4: Initial Affordance Calculation ==========
    affordance_masks, grasp_angles, widths, major_axes, minor_axes = compute_initial_affordance(
        image_rgb, masks, sam_centers
    )

    # ========== Step 5: Occlusion Detection ==========
    occlusion_info = detect_all_occlusions(depth_image, masks, affordance_masks, sam_centers, mclass_ids, class_names)

    # ========== Step 6: Mutual Occlusion Handling ==========
    mutual_pairs = find_mutual_occlusions(occlusion_info)
    if mutual_pairs:
        print(f"[INFO] Mutual occlusions detected: {mutual_pairs}. Refining affordance...")
        affordance_masks, grasp_angles, widths, major_axes, minor_axes = recompute_for_mutual_occlusions(
            image_rgb, depth_image, masks, sam_centers, mclass_ids, class_names,
            mutual_pairs, major_axes, minor_axes, affordance_masks, grasp_angles, widths
        )
        sam_centers = get_mask_centers(affordance_masks)
    
    graspable_flags, grasp_difficulties = compute_grasp_difficulty(masks, affordance_masks)

    # ========== Step 7: 3D Projection & Coordinate Transformation ==========
    centers_3d = get_3d_coords_from_centers(npz_path, sam_centers, depth_intrin)
    code, pos = arm.get_position(is_radian=False)
    tcp_pose = pos[:6]
    world_coords = []
    
    for i, pt in enumerate(centers_3d):
        if pt is None or np.allclose(pt, [0, 0, 0]):
            print(f"[WARN] Object {i} has invalid 3D point, skipping.")
            world_coords.append(None)
            continue
        world_point = camera_to_world(np.array(pt), tcp_pose, min_z_mm=225, verbose=False)
        world_coords.append(world_point.tolist())

    # ========== Step 8: Save Scene Info to JSON ==========
    save_scene_info_to_json(
        class_ids=mclass_ids,
        coords_3d=world_coords,
        output_path=json_path,
        yolo_path=yolo_model_path,
        widths=widths,
        angles=grasp_angles,
        quaternions=None,
        occlusion_info=occlusion_info,
        graspable_flags=graspable_flags
    )
    print(f"[INFO] Scene information saved to {json_path}.")

    # ========== Step 9: LLM Planning ==========
    cmd = wait_for_command(timeout_sec=10)
    if not cmd:
        try:
            cmd = input("Enter your robot instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] No input received. Program terminating.")
            sys.exit(0)
    
    if not cmd:
        print("[ERROR] Empty instruction. Program terminating.")
        sys.exit(0)

    print(f"[INFO] Planning subtasks for: '{cmd}'")
    task_description = subtasks_planner(cmd)
    
    subtasks_file = "llm_subtasks.txt"
    with open(subtasks_file, "w") as f:
        f.write(task_description)

    # ========== Step 10: Trajectory Execution ==========
    trajectory_json_path = "trajectory_plan.json"
    trajectory_plan_on_real(
        subtask_txt_path=subtasks_file,
        json_path=json_path,
        save_json_path=trajectory_json_path
    )
    print("[INFO] Executing trajectory...")
    execute(trajectory_json_path, arm)
    print("[INFO] Task completed.")