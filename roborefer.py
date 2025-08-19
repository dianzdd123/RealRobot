# Import utility functions
from utils.sam2_utils import segment_with_sam2_points
from utils.coordinate import get_3d_coords_from_centers
from utils.save_json import save_scene_info_to_json
from utils.rotate import extract_grasp_infos
from xarm.wrapper import XArmAPI
from utils.API import subtasks_planner, trajectory_plan_roborefer, roborefer_prompts
from utils.camera_to_world import camera_to_world
from utils.traj_execute import execute
from utils.roborefer_test import start_server, run_roborefer

# Import standard libraries
import pyrealsense2 as rs
import os
import cv2, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

if __name__ == "__main__": 
    # Paths and model configuration
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    # path_base = "imgs/3"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"
    save_path = "imgs/roborefer_point.png"

    # ========== Step 1: Robot Initialization and Image Capture ==========
    arm = XArmAPI('192.168.1.117')  # Connect to robot
    arm.motion_enable(enable=True)
    arm.set_state(0)     # Set to ready state
    arm.set_mode(0)      # Set to position control mode

    # Move robot to initial viewing position
    # arm.set_position(x=0, y=-450, z=550, roll=180, pitch=0, yaw=0, speed=65, wait=True)
    arm.set_position(x=0, y=-346, z=500, roll=180, pitch=0, yaw=0, speed=80, wait=True)
    arm.set_gripper_position(850, wait=True)

    # Start RealSense camera stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    # Warm up the camera
    for _ in range(10):
        frames = pipeline.wait_for_frames()
    # Capture one aligned RGB-D frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        print("[ERROR] Failed to capture frames.")
        exit(1)
    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    # Obtain camera intrin
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # Save RGB and depth images
    image_rgb = cv2.imwrite(img_path, color_image)
    np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    pipeline.stop()  # Stop the camera stream

    # ========== Step 2: Parse instruction and get RoboRefer results==========
    user_cmd = "grasp corn"
    # user_cmd = input("Enter your robot instruction: ")
    prompts = roborefer_prompts(user_cmd)

    server = start_server()
    # 获取每个语义目标的像素点坐标（u, v）
    points_uv, descriptions = run_roborefer(
        image_path=img_path,
        prompts=prompts,
        server=server,
        verbose=True,
        save_path=save_path,
        return_name=True
    )
    server.terminate()
    masks, sam_centers, _ = segment_with_sam2_points(
        image_rgb = image_rgb,
        points_uv = points_uv,
        sam_model_path = sam2_model_path,
        input_image_path = img_path,
        point_radius_px=32,
        prefer_points_prompt=True
    )
    
    grasp_infos, grasp_img_path = extract_grasp_infos(
        image_rgb=image_rgb,
        masks=masks,
        pixel_to_meter=0.0005,
        output_path=f"{path_base}_grasp_visual.png"
    )
    # ========== Step 4: 3D Projection and Camera to World Coordinate==========
    centers_3d = get_3d_coords_from_centers(npz_path, sam_centers, depth_intrin)  # 约定：返回米
    code, pos = arm.get_position(is_radian=False)
    tcp_pose = pos[:6]  # [x_mm, y_mm, z_mm, roll, pitch, yaw_deg]

    world_coords = []
    for i, pt in enumerate(centers_3d):
        # 统一与代码2相同的健壮性判断
        if pt is None or (hasattr(pt, "__len__") and all(abs(v) < 1e-9 for v in pt)):
            print(f"[WARN] Object {i} has invalid 3D point, skipping")
            world_coords.append(None)
            continue

        camera_point_m = np.asarray(pt, dtype=float)  # already in meters
        world_point = camera_to_world(
            camera_point_m,
            tcp_pose,
            verbose=True  # optionalgra
        )
        world_coords.append(world_point.tolist())

    # ========== Step 5: Save to JSON ==========
    output_data = []
    for name, (x2d, y2d), cam3d, world3d in zip(descriptions, points_uv, centers_3d, world_coords):
        if x2d is None or y2d is None:
            continue
        output_data.append({
            "name": name,
            "2d_coord": [round(x2d, 2), round(y2d, 2)],
            "3d_camera": [round(float(v), 2) for v in cam3d],
            "3d_world": [round(float(v), 2) for v in world3d]
        })

    json_path = "grasp_infos.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ========== Step 7: Parse User Command with LLM ==========
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    task_description = subtasks_planner(user_cmd)
    subtasks_file = "llm_subtasks.txt"
    # Save parsed sub-tasks
    with open(subtasks_file, "w") as f:
        f.write(task_description)

    # ========== Step 8: Execute Trajectory Plan ==========
    trajectory_jaon_path = "trajectory_plan.json"
    trajectory_plan_roborefer(subtask_txt_path=subtasks_file, json_path=json_path, save_json_path=trajectory_jaon_path)
    # execute(trajectory_jaon_path, arm)