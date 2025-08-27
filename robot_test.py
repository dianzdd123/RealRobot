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

# Import standard libraries
import pyrealsense2 as rs
import os
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

def init_robot_and_camera(robot_ip="192.168.1.117", img_path="color.png", npz_path="frames.npz"):
    # ========== Step 1: Robot Initialization ==========
    arm = XArmAPI(robot_ip)
    arm.motion_enable(enable=True)
    arm.set_state(0)     # Set to ready state
    arm.set_mode(0)      # Set to position control mode

    # Move robot to initial viewing position
    arm.set_position(x=0, y=-346, z=500, roll=180, pitch=0, yaw=0, speed=80, wait=True)
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
        raise RuntimeError("[ERROR] Failed to capture frames.")

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

if __name__ == "__main__": 
    '''
    System Workflow:
    1. Initialize robot and capture an RGB-D image of the desktop using RealSense
    2. Run YOLO to detect objects and their bounding boxes
    3. Use SAM2 to segment each object and compute grasp angle and width
    4. Convert user natural language command into sub-tasks using LLM
    5. Project image centers into 3D space (camera coordinates)
    6. Transform 3D camera coordinates into world coordinates using robot pose
    7. Save results into JSON format
    8. Use LLM to generate trajectory plan based on the JSON and command
    9. Execute the trajectory using the robot
    '''

    # Paths and model configuration
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"

    # ========== Step 1: Robot Initialization and Image Capture ==========
    arm, image_rgb, depth_image, depth_intrin = init_robot_and_camera(img_path=img_path, npz_path=npz_path)
    print("Robot and camera initialized successfully.")

    # ========== Step 2: Object Detection with YOLO and SAM2==========
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
    
    if len(boxes) == 0:
        print("No objects detected by YOLO. Exit!")
        exit()

    mboxes, mclass_ids, mcenters, groups = merge_detections_by_depth_and_proximity(
        boxes=boxes,
        class_ids=class_ids,
        centers_xy=yolo_centers,                 # 使用 YOLO 返回的中心
        depth_image=depth_image.astype(np.uint16),
        pixel_dist_thresh=100,
        depth_diff_thresh=20,
        same_class_only=True,
        win=3
    )
    
    # ========== Step 3: Segmentation and Grasp Info ==========
    masks, sam_centers, _ = segment_with_sam2(image_rgb=image_rgb, boxes=mboxes, sam_model_path=sam2_model_path, input_image_path=img_path)

    # ========== Step 3: Caculate the angle and width of object==========
    grasp_infos, grasp_img_path = extract_grasp_infos(image_rgb=image_rgb, masks=masks, pixel_to_meter=0.0005, output_path=f"{path_base}_grasp_visual.png")
    
    # ========== Step 4: 3D Projection and Camera to World Coordinate==========
    centers_3d = get_3d_coords_from_centers(npz_path, sam_centers, depth_intrin)
    code, pos = arm.get_position(is_radian=False)
    tcp_pose = pos[:6]  # [x, y, z, roll, pitch, yaw]
    world_coords = []

    for i, pt in enumerate(centers_3d):
        if pt is None or np.allclose(pt, [0, 0, 0]):
            print(f"[WARN] Object {i} has invalid 3D point, skipping")
            world_coords.append(None)
            continue

        camera_point_m = np.array(pt)  # already in meters
        world_point = camera_to_world(
            camera_point_m,
            tcp_pose,
            verbose=True  # optionalgra
        )
        world_coords.append(world_point.tolist())

    # ========== Step 6: Save to JSON ==========
    save_scene_info_to_json(
        # class id and name
        class_ids=class_ids,
        # 3D center
        coords_3d=world_coords,
        output_path=json_path,
        # YOLO path for id and name match
        yolo_path=yolo_model_path,
        widths=[info["width"] for info in grasp_infos],
        angles=[info["angle"] for info in grasp_infos],
    )
    # print(grasp_infos)
    # ========== Step 7: Parse User Command with LLM ==========
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_input = input("Enter your robot instruction: ")
    task_description = subtasks_planner(user_input)
    subtasks_file = "llm_subtasks.txt"
    # Save parsed sub-tasks
    with open(subtasks_file, "w") as f:
        f.write(task_description)

    # ========== Step 8: Execute Trajectory Plan ==========
    trajectory_jaon_path = "trajectory_plan.json"
    trajectory_plan_on_real(subtask_txt_path=subtasks_file, json_path=json_path, save_json_path=trajectory_jaon_path)
    execute(trajectory_jaon_path, arm)


    
