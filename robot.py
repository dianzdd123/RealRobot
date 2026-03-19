"""
Standard Robotic Grasping Pipeline
Focuses on object detection, segmentation, and trajectory execution without occlusion reasoning.
"""

# Import utility functions
from utils.yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity                                                                                
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import get_3d_coords_from_centers
from utils.save_json import save_json_from_detection, save_scene_info_to_json
from utils.rotate import extract_grasp_infos
from xarm.wrapper import XArmAPI
from utils.API import subtasks_planner, trajectory_plan_on_real
from utils.camera_to_world import camera_to_world
from utils.traj_execute import execute
from utils.voice_hybrid import wait_for_command

# Import standard libraries
import pyrealsense2 as rs                                                            
import os
import cv2
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

def init_robot_and_camera(robot_ip="192.168.1.117", img_path="color.png", npz_path="frames.npz"):
    """
    Initializes the xArm robot and Intel RealSense camera.
    
    Args:
        robot_ip (str): IP address of the xArm controller.
        img_path (str): Path to save the captured RGB image.
        npz_path (str): Path to save the compressed RGB-D data.
        
    Returns:
        tuple: (arm_instance, color_image, depth_image, depth_intrinsics)
    """
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

    # Warm up the camera sensor
    for _ in range(10):
        pipeline.wait_for_frames()

    # Capture aligned RGB-D frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        raise RuntimeError("[ERROR] Failed to capture frames from RealSense.")

    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Obtain camera intrinsics for 3D projection
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    # Save data for offline reference/debugging
    cv2.imwrite(img_path, color_image)
    np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    pipeline.stop()

    return arm, color_image, depth_image, depth_intrin

if __name__ == "__main__": 
    # ========== Configuration ==========
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"

    # ========== Step 1: System Initialization ==========
    arm, image_rgb, depth_image, depth_intrin = init_robot_and_camera(img_path=img_path, npz_path=npz_path)
    print("[INFO] Robot and camera initialized successfully.")

    # ========== Step 2: Object Detection (YOLO) ==========
    image_rgb, boxes, yolo_centers, class_ids, _ = detect_fruits(img_path, yolo_model_path)
    
    if len(boxes) == 0:
        print("[ERROR] No objects detected by YOLO. Exiting.")
        sys.exit(0)

    # Merge overlapping detections based on depth and proximity
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
    
    # ========== Step 3: Segmentation & Grasp Analysis (SAM2) ==========
    masks, sam_centers, _ = segment_with_sam2(
        image_rgb=image_rgb, 
        boxes=mboxes, 
        sam_model_path=sam2_model_path, 
        input_image_path=img_path
    )

    # Calculate grasp angle and gripper width for each object
    grasp_infos, _ = extract_grasp_infos(
        image_rgb=image_rgb, 
        masks=masks, 
        pixel_to_meter=0.0005, 
        output_path=f"{path_base}_grasp_visual.png"
    )
    
    # ========== Step 4: 3D Mapping & Coordinate Transformation ==========
    centers_3d = get_3d_coords_from_centers(npz_path, sam_centers, depth_intrin)
    _, pos = arm.get_position(is_radian=False)
    tcp_pose = pos[:6]  # [x, y, z, roll, pitch, yaw]
    world_coords = []

    for i, pt in enumerate(centers_3d):
        if pt is None or np.allclose(pt, [0, 0, 0]):
            print(f"[WARN] Object {i} has invalid 3D point, skipping.")
            world_coords.append(None)
            continue

        # Transform camera coordinates (meters) to world coordinates (mm)
        world_point = camera_to_world(
            np.array(pt),
            tcp_pose,
            min_z_mm=220,
            verbose=False
        )
        world_coords.append(world_point.tolist())

    # ========== Step 5: Scene Serialization ==========
    save_scene_info_to_json(
        class_ids=class_ids,
        coords_3d=world_coords,
        output_path=json_path,
        yolo_path=yolo_model_path,
        widths=[info["width"] for info in grasp_infos],
        angles=[info["angle"] for info in grasp_infos],
    )
    print(f"[INFO] Scene information serialized to {json_path}.")

    # ========== Step 6: Task Planning (LLM) ==========
    cmd = wait_for_command(timeout_sec=10)

    if not cmd:
        try:
            cmd = input("Enter your robot instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] User cancelled input. Exiting.")
            sys.exit(0)

    if not cmd:
        print("[ERROR] Empty instruction received. Exiting.")
        sys.exit(0)

    print(f"[INFO] Planning subtasks for: '{cmd}'")
    task_description = subtasks_planner(cmd)

    # Save planned steps for execution
    subtasks_file = "llm_subtasks.txt"
    with open(subtasks_file, "w") as f:
        f.write(task_description)

    # ========== Step 7: Trajectory Execution ==========
    trajectory_json_path = "trajectory_plan.json"
    trajectory_plan_on_real(
        subtask_txt_path=subtasks_file, 
        json_path=json_path, 
        save_json_path=trajectory_json_path
    )
    
    print("[INFO] Executing planned trajectory...")
    execute(trajectory_json_path, arm)
    print("[INFO] Task successfully completed.")