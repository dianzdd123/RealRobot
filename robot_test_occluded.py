# Import utility functions
from utils.yolo_utils import detect_fruits
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import get_3d_coords_from_centers
from utils.save_json import save_json_from_detection, save_scene_info_to_json
from utils.rotate import extract_grasp_infos, infer_grasp_angle_and_width, compute_affordance_mask
from xarm.wrapper import XArmAPI
from utils.API import subtasks_planner, trajectory_plan_on_real2
from utils.camera_to_world import camera_to_world
from utils.traj_execute import execute
from occluded import (
    check_occlusion,
    build_occlusion_graph,
    get_grasp_sequence,
    compute_grasp_difficulty,
    draw_mask_boundaries,
    detect_ambiguity,
    summarize_occlusion_graph,
)

# Import standard libraries
import pyrealsense2 as rs
import os
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

def robust_center_depth(depth_image, cx, cy, win=5):
    """
    从深度图中取 (cx, cy) 周围 win×win 的中位数作中心深度，忽略 0（洞值）。
    返回 int 深度，若无有效值返回 None
    """
    h, w = depth_image.shape
    cx = int(round(cx)); cy = int(round(cy))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    x0, x1 = max(cx - win, 0), min(cx + win + 1, w)
    y0, y1 = max(cy - win, 0), min(cy + win + 1, h)
    patch = depth_image[y0:y1, x0:x1].astype(np.uint16)
    vals = patch[patch > 0]
    if vals.size == 0:
        return None
    return int(np.median(vals))

def depths_from_centers(depth_image, centers_xy, win=5):
    """对一组中心点批量取鲁棒深度；返回 List[int or None]"""
    return [robust_center_depth(depth_image, cx, cy, win=win) for (cx, cy) in centers_xy]

def merge_detections_by_depth_and_proximity(
    boxes, class_ids, centers_xy, depth_image,
    pixel_dist_thresh=60, depth_diff_thresh=20, same_class_only=True, win=3
):
    """
    - 两目标中心像素距离 < pixel_dist_thresh
    - 中心深度差 < depth_diff_thresh（鲁棒中位数）
    - (可选) 同类别
    视为同一物体，合并为联合 bbox。
    返回: merged_boxes, merged_class_ids, merged_centers, groups(原索引组)
    """
    N = len(boxes)
    used = [False] * N
    groups = []

    # 预取每个中心的鲁棒深度
    center_depths = depths_from_centers(depth_image, centers_xy, win=win)

    for i in range(N):
        if used[i]:
            continue
        group = [i]; used[i] = True
        xi, yi = centers_xy[i]; di = center_depths[i]; ci = class_ids[i]

        for j in range(i + 1, N):
            if used[j]:
                continue
            xj, yj = centers_xy[j]; dj = center_depths[j]; cj = class_ids[j]

            if same_class_only and (ci != cj):
                continue
            if di is None or dj is None:
                continue  # 没有可靠深度时跳过（也可放宽只用像素距离）

            # 像素距离与深度差阈值
            dist = np.hypot(xi - xj, yi - yj)
            if dist > pixel_dist_thresh:
                continue
            if abs(di - dj) > depth_diff_thresh:
                continue

            group.append(j); used[j] = True

        groups.append(group)

    # 合并每个组
    merged_boxes, merged_class_ids, merged_centers = [], [], []
    for g in groups:
        xs1, ys1, xs2, ys2 = [], [], [], []
        cls_list, cx_list, cy_list = [], [], []
        for k in g:
            x1, y1, x2, y2 = boxes[k]
            xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
            cls_list.append(class_ids[k])
            cx, cy = centers_xy[k]
            cx_list.append(cx); cy_list.append(cy)

        # 合并框：取 min/max
        mx1, my1, mx2, my2 = min(xs1), min(ys1), max(xs2), max(ys2)
        merged_boxes.append([mx1, my1, mx2, my2])

        # 类别取众数（并列取第一个）
        vals, counts = np.unique(cls_list, return_counts=True)
        merged_class_ids.append(int(vals[np.argmax(counts)]))

        # 新中心：平均值（或用合并框中心）
        merged_centers.append((int(np.mean(cx_list)), int(np.mean(cy_list))))

    return merged_boxes, merged_class_ids, merged_centers, groups

if __name__ == "__main__": 
    # Paths and model configuration
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"

    # ========== Step 1: Robot Initialization and Image Capture ==========
    arm = XArmAPI('192.168.1.117')  # Connect to robot

    arm.motion_enable(enable=True)
    arm.set_state(0)     # Set to ready state
    arm.set_mode(0)      # Set to position control mode

    # Move robot to initial viewing position
    # arm.set_position(x=221.7, y=-3.4, z=450, roll=180, pitch=0, yaw=0, speed=65, wait=True)
    # arm.set_position(x=0, y=-450, z=550, roll=180, pitch=0, yaw=0, speed=65, wait=True)
    arm.set_position(x=0, y=-346, z=500, roll=180, pitch=0, yaw=0, speed=65, wait=True)
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
    cv2.imwrite(img_path, color_image)
    np.savez_compressed(npz_path, color=color_image, depth=depth_image)

    pipeline.stop()  # Stop the camera stream

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
        pixel_dist_thresh=60,
        depth_diff_thresh=20,
        same_class_only=True,
        win=3
    )
    print(f"[合并] 从 {len(boxes)} -> {len(mboxes)}，合并组: {groups}")

    # ========== Step 3: Segmentation and Grasp Info ==========
    masks, sam_centers, result_path = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=mboxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )
    # ========== Step 3: Caculate the angle and width of object==========
    grasp_infos, grasp_img_path = extract_grasp_infos(
        image_rgb=image_rgb,
        masks=masks,
        pixel_to_meter=0.0005,
        output_path=f"{path_base}_grasp_visual.png"
    )
    affordance_masks = []

    for i, m in enumerate(masks):
        # 计算抓取角度和宽度，同时返回主副轴
        grasp_angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(
            m, draw_on=image_rgb, object_index=i
        )
        mean = np.array(sam_centers[i], dtype=np.float32)

        aff_mask = compute_affordance_mask(
            mask=m,
            major_axis=major_axis,
            minor_axis=minor_axis,
            mean=mean,
            center_ratio=0.4,
            expand_pixels=15
        )
        affordance_masks.append(aff_mask)

    # 遮挡检测
    occlusion_line_pairs = []
    for idx, mask in enumerate(masks):
        center = sam_centers[idx]
        is_occluded, contact_pair = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=affordance_masks[idx],
            center=center,
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            pixel_gap=5,
            debug=False
        )
        if is_occluded and contact_pair:
            occlusion_line_pairs.append(contact_pair)

    # 可视化遮挡关系
    vis_img = draw_mask_boundaries(
        image_rgb,
        masks,
        ids=list(range(len(masks))),
        line_pairs=occlusion_line_pairs
    )
    cv2.imwrite("imgs/occluded.png", vis_img)
    print("[INFO] 遮挡图像已保存至 imgs/occluded.png")

    # 构建遮挡图与打印分析
    occlusion_graph, _ = build_occlusion_graph(
        masks=masks,
        depth_image=depth_image,
        sam_centers=sam_centers,
        class_ids=class_ids,
        pixel_gap=20,
        debug=False
    )

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
    trajectory_plan_on_real2(subtask_txt_path=subtasks_file, json_path=json_path, save_json_path=trajectory_jaon_path)
    execute(trajectory_jaon_path, arm)


    
