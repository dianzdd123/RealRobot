import numpy as np
import cv2
import os
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from utils.sam2_utils import segment_with_sam2
from utils.rotate import compute_affordance_mask
from utils.rotate import draw_affordance_on_image
from utils.rotate import infer_grasp_angle_and_width

def robust_center_depth(depth_image, cx, cy, win=5):
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

def check_occlusion(depth_image, mask, affordance_mask, center, all_masks=None, all_affordance_masks=None, pixel_gap=5, debug=False):
    """
    遮挡判断逻辑（更新版）：
    1）深度遮挡：边界深度 < 中心深度（中心被遮）
    2）靠近遮挡：其他物体靠近当前aff区域 且 深度比当前小或相近（才算遮挡）

    返回:
        is_occluded: True / False
        contact_pair: (pt1, pt2) if valid
    """
    x, y = center
    h, w = depth_image.shape
    if not (0 <= x < w and 0 <= y < h):
        return False, None

    center_depth = robust_center_depth(depth_image, x, y, win=5)
    if center_depth is None:
        return False, None

    # === 1) 深度遮挡判断 ===
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) == 0:
        return False, None
    boundary_points = contours[0][:, 0, :]

    boundary_depths = [robust_center_depth(depth_image, pt[0], pt[1], win=3)
                       for pt in boundary_points if 0 <= pt[0] < w and 0 <= pt[1] < h]
    boundary_depths = [d for d in boundary_depths if d is not None]

    if boundary_depths:
        min_boundary_depth = np.min(boundary_depths)
        depth_diff_threshold = 10  # mm
        if center_depth - min_boundary_depth > depth_diff_threshold:
            if debug:
                print(f"[遮挡-深度] 中心: {center_depth}, 边界最小: {min_boundary_depth}, 差: {center_depth - min_boundary_depth}")
            return True, None

    # === 2) 靠近遮挡判断（aff vs 他物边界） ===
    if all_masks and all_affordance_masks:
        aff_contours, _ = cv2.findContours(affordance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not aff_contours:
            return False, None
        aff_boundary = aff_contours[0][:, 0, :]  # (N, 2)

        for other_mask, other_aff in zip(all_masks, all_affordance_masks):
            if np.array_equal(other_mask, mask):
                continue  # 跳过自己

            other_contours, _ = cv2.findContours(other_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not other_contours:
                continue
            other_boundary = other_contours[0][:, 0, :]

            dists = np.linalg.norm(aff_boundary[:, None, :] - other_boundary[None, :, :], axis=2)
            min_dist = np.min(dists)
            min_pos = np.unravel_index(np.argmin(dists), dists.shape)

            if min_dist < pixel_gap:
                # pt1 = tuple(aff_boundary[min_pos[0]])     # 我的抓取区域边界
                # pt2 = tuple(other_boundary[min_pos[1]])   # 对方边界
                # d_aff = robust_center_depth(depth_image, pt1[0], pt1[1], win=3)
                # d_other = robust_center_depth(depth_image, pt2[0], pt2[1], win=3)

                # if d_aff is None or d_other is None:
                #     continue

                # depth_margin = 0  # mm 容忍范围（同一平面也算遮挡）
                # if d_aff < d_other - depth_margin:
                #     if debug:
                #         print(f"[无遮挡] 当前:{d_aff} 在前，对方:{d_other} 更远 → 不遮挡")
                #     return False, None
                # else:
                #     if debug:
                #         print(f"[靠近遮挡] 当前:{d_aff} 不明显在前，对方:{d_other} 靠近或在前 → 遮挡成立")
                #     return True, (pt1, pt2)
                return True

    return False, None

def build_occlusion_graph(masks, depth_image, sam_centers, class_ids,
                          affordance_masks=None, pixel_gap=20, debug=False):
    """
    构建遮挡图，基于 check_occlusion 判断是否被遮挡。
    返回 occlusion_graph 和 遮挡边 contact_pair。
    """
    occlusion_graph = {i: [] for i in range(len(masks))}
    occlusion_edges = []

    for i in range(len(masks)):
        mask = masks[i]
        center = sam_centers[i]
        aff_mask = affordance_masks[i] if affordance_masks else None

        is_occluded, contact_pair = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=aff_mask,
            center=center,
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            pixel_gap=pixel_gap,
            debug=debug
        )

        if is_occluded and contact_pair is not None:
            pt_other = contact_pair[1]  # 是谁遮挡了我

            for j, other_mask in enumerate(masks):
                if j == i:
                    continue
                if 0 <= pt_other[0] < depth_image.shape[1] and 0 <= pt_other[1] < depth_image.shape[0]:
                    if other_mask[pt_other[1], pt_other[0]] > 0:
                        occlusion_graph[i].append(j)
                        dist = np.linalg.norm(np.array(contact_pair[0]) - np.array(contact_pair[1]))
                        occlusion_edges.append(((i, j), dist))
                        if debug:
                            print(f"[图构建] 物体 {i} 被 {j} 遮挡，最近距离: {dist:.1f}px")
                        break

    return occlusion_graph, occlusion_edges

def get_grasp_sequence(graph):
    visited = set()
    visiting = set()  # 当前路径上的节点，用于检测循环
    sequence = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        if node in visited:
            return
        if node in visiting:
            has_cycle = True
            print(f"[警告] 检测到图中存在环路，节点 {node} 在路径中重复访问")
            return
        visiting.add(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        visiting.remove(node)
        visited.add(node)
        sequence.append(node)

    for node in graph:
        dfs(node)

    if has_cycle:
        print("[警告] 遮挡图存在有向环，抓取顺序可能不准确")

    return sequence[::-1]

def compute_grasp_difficulty(graph):
    def dfs_depth(node, visited):
        if node in visited:
            return 0  # 避免循环导致爆栈
        visited.add(node)
        if not graph.get(node):
            return 0
        max_child_depth = 0
        for n in graph[node]:
            max_child_depth = max(max_child_depth, dfs_depth(n, visited.copy()))
        return 1 + max_child_depth

    difficulty_map = {}
    for node in graph:
        depth = dfs_depth(node, set())
        if depth == 0:
            difficulty = "Easy"
        elif depth == 1:
            difficulty = "Medium"
        else:
            difficulty = "Hard"
        difficulty_map[node] = difficulty

    return difficulty_map

def detect_ambiguity(class_ids):
    from collections import Counter
    counts = Counter(class_ids)
    return any(v > 1 for v in counts.values())

def summarize_occlusion_graph(occlusion_graph, class_ids, class_names):
    print("\n=== 遮挡图分析结果 ===")
    difficulty_map = compute_grasp_difficulty(occlusion_graph)
    sequence = get_grasp_sequence(occlusion_graph)
    ambiguous = detect_ambiguity(class_ids)

    for i in range(len(class_ids)):
        blockers = occlusion_graph.get(i, [])
        blocker_names = [class_names[class_ids[j]] for j in blockers]
        print(f"物体 {i} ({class_names[class_ids[i]]}) 被遮挡者: {blocker_names if blocker_names else '无遮挡'}，难度: {difficulty_map[i]}")

    print("\n抓取执行顺序（优先抓取遮挡层最深的物体）:")
    for idx in sequence:
        print(f" -> 物体 {idx} ({class_names[class_ids[idx]]})")

    print(f"\n场景中是否存在歧义物体（相同类别多个实例）: {'是' if ambiguous else '否'}")

def draw_mask_boundaries(image, masks, color=(0, 255, 0), thickness=2, ids=None, line_pairs=None):
    vis_image = image.copy()
    for idx, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(vis_image, contours, -1, color, thickness)
        if ids is not None:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(vis_image, str(ids[idx]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 添加红色连接线
    if line_pairs:
        for pt1, pt2 in line_pairs:
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)
    return vis_image

if __name__ == "__main__":
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    # path_base = "imgs/img_from_camera"
    path_base = "test_imgs/img4"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"

    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names
    '''
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
    # Save RGB and depth images
    success = cv2.imwrite(img_path, color_image)
    if not success:
        print("[ERROR] RGB图像保存失败！")

    np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    pipeline.stop()
    
    # data = np.load(npz_path)
    # color_image = data["color"]
    # depth_image = data["depth"]
    '''
    if not os.path.exists(img_path):
        print(f"[ERROR] 图像路径不存在: {img_path}")
        exit(1)
    color_image = cv2.imread(img_path)
    data = np.load(npz_path)
    depth_image = data["depth"]

    # 目标检测
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
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
    
    # SAM 分割
    masks, sam_centers, result_path = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=mboxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )
    # 计算获得 affordance area
    image_with_aff = image_rgb.copy()
    affordance_masks = []
    for i, m in enumerate(masks):
        # 得到主副轴和质心
        grasp_angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)

        mean = np.array(sam_centers[i], dtype=np.float32)
        # 基于主副轴和质心生成 affordance mask（相对原宽度固定像素外扩）
        aff_mask = compute_affordance_mask(
            mask=m,
            major_axis=major_axis,
            minor_axis=minor_axis,
            mean=mean,
            center_ratio=0.4,
            expand_pixels=20
        )

        affordance_masks.append(aff_mask)

        # 绘制叠加
        image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask, color=(0, 255, 0), alpha=0.4)
    # 保存叠加可视化结果
    cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))
    print("图像已保存至 output_affordance_overlay.png")

    # 遮挡检测
    occlusion_line_pairs = []
    for idx, mask in enumerate(masks):
        class_id = class_ids[idx]
        class_name = class_names.get(class_id, f"class_{class_id}")
        center = sam_centers[idx]

        is_occluded, contact_pair = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=affordance_masks[idx],
            center=center,
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            pixel_gap=5,
            debug=True
        )

        if is_occluded:
            print(f"物体 {idx} ({class_name}) 被遮挡")
            if contact_pair:
                occlusion_line_pairs.append(contact_pair)
        else:
            print(f"物体 {idx} ({class_name})无遮挡")

    # 可视化：掩码边界 + 最短距离红线
    vis_img = draw_mask_boundaries(
        image_rgb,
        masks,
        ids=list(range(len(masks))),
        line_pairs=occlusion_line_pairs
    )
    cv2.imwrite("imgs/occluded.png", vis_img)

    # occlusion_graph, _ = build_occlusion_graph(
    #     masks=masks,
    #     depth_image=depth_image,
    #     sam_centers=sam_centers,
    #     class_ids=class_ids,
    #     pixel_gap=20,
    #     debug=False
    # )

    # # 打印分析结果
    # summarize_occlusion_graph(
    #     occlusion_graph=occlusion_graph,
    #     class_ids=class_ids,
    #     class_names=class_names
    # )