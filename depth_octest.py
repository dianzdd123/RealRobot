import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from utils.sam2_utils import segment_with_sam2
from utils.rotate import compute_affordance_mask, draw_affordance_on_image, infer_grasp_angle_and_width

def robust_center_depth(depth_image, cx, cy, win=5):
    h, w = depth_image.shape
    cx = int(round(cx)); cy = int(round(cy))
    x0, x1 = max(cx - win, 0), min(cx + win + 1, w)
    y0, y1 = max(cy - win, 0), min(cy + win + 1, h)
    patch = depth_image[y0:y1, x0:x1].astype(np.uint16)
    vals = patch[patch > 0]
    return int(np.median(vals)) if vals.size else None

def check_occlusions(masks, affordance_masks, sam_centers, depth_image,
                         class_ids, class_names,
                         depth_threshold=20, aff_threshold=10, iterations=1,
                         verbose=True):
    """
    综合遮挡检测：同时判断深度遮挡和靠近遮挡。
    返回：dict，key 为遮挡类型，value 为 (被遮挡索引, 遮挡索引) 列表。
    """
    h, w = depth_image.shape
    kernel = np.ones((3, 3), np.uint8)
    dilated_masks = [cv2.dilate(m.astype(np.uint8), kernel, iterations=iterations) for m in masks]

    depth_occlusion_pairs = set()
    proximity_occlusion_pairs = set()

    for i in range(len(masks)):
        center_i = sam_centers[i]
        d_i = robust_center_depth(depth_image, int(center_i[0]), int(center_i[1]), win=5)
        if d_i is None:
            continue

        for j in range(len(masks)):
            if i == j:
                continue

            # --------- 条件 1：边界深度遮挡 ----------
            inter = cv2.bitwise_and(dilated_masks[i], dilated_masks[j])
            if np.sum(inter) > 0:
                center_j = sam_centers[j]
                d_j = robust_center_depth(depth_image, int(center_j[0]), int(center_j[1]), win=5)
                if d_j is not None and d_j + depth_threshold < d_i:
                    if (i, j) not in depth_occlusion_pairs:
                        if verbose:
                            name_i = class_names[class_ids[i]]
                            name_j = class_names[class_ids[j]]
                            print(f"[Depth Occlusion] Object {i} ({name_i}) is occluded by {j} ({name_j})")
                            print(f" - Depth diff: {d_i} - {d_j} = {d_i - d_j}")
                        depth_occlusion_pairs.add((i, j))

            # --------- 条件 2：affordance 区域靠近遮挡 ----------
            aff_mask = affordance_masks[i]
            overlap = cv2.bitwise_and(aff_mask.astype(np.uint8), masks[j].astype(np.uint8))
            if np.sum(overlap) > 0:
                ys, xs = np.where(overlap > 0)
                sampled_depths = depth_image[ys, xs]
                valid_depths = sampled_depths[(sampled_depths > 0) & (sampled_depths < 10000)]
                if valid_depths.size > 0:
                    min_depth_j = np.min(valid_depths)
                    if min_depth_j + aff_threshold < d_i:
                        if (i, j) not in proximity_occlusion_pairs:
                            if verbose:
                                name_i = class_names[class_ids[i]]
                                name_j = class_names[class_ids[j]]
                                print(f"[Proximity Occlusion] Object {i} ({name_i}) is occluded by {j} ({name_j}) on affordance area.")
                                print(f" - depth_i: {d_i}, min_depth_j: {min_depth_j}, Δ={d_i - min_depth_j}")
                            proximity_occlusion_pairs.add((i, j))

    return {
        "depth_occlusion": list(depth_occlusion_pairs),
        "proximity_occlusion": list(proximity_occlusion_pairs)
    }

def check_occlusion_by_intersection_and_depth(masks, sam_centers, depth_image,
                                               class_ids, class_names,
                                               depth_threshold=15, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    dilated_masks = [cv2.dilate(m.astype(np.uint8), kernel, iterations=iterations) for m in masks]

    occlusion_pairs = []

    for i in range(len(masks)):
        for j in range(len(masks)):
            if i == j:
                continue
            inter = cv2.bitwise_and(dilated_masks[i], dilated_masks[j])
            if np.sum(inter) == 0:
                continue  # no intersection

            # 中心深度比较
            c_i = sam_centers[i]
            c_j = sam_centers[j]
            d_i = robust_center_depth(depth_image, int(c_i[0]), int(c_i[1]), win=5)
            d_j = robust_center_depth(depth_image, int(c_j[0]), int(c_j[1]), win=5)
            if d_i is None or d_j is None:
                continue

            # 判断谁在前面
            if d_j + depth_threshold < d_i:
                name_i = class_names[class_ids[i]]
                name_j = class_names[class_ids[j]]
                print(f"[Occlusion] Object {i} ({name_i}) is occluded by object {j} ({name_j})")
                print(f" - Depth diff: {d_i} - {d_j} = {d_i - d_j}")
                occlusion_pairs.append((i, j))
    return occlusion_pairs

def draw_mask_intersections_expanded(image, masks, iterations=3, save_path="expanded_mask_overlap.png"):
    """
    膨胀每个 mask 并绘制边界，若边界相交则高亮标记。
    - masks: List[np.ndarray], 每个 mask 是 HxW 的二值图像
    - iterations: 膨胀像素数量（越大 mask 越大）
    - save_path: 保存图像路径
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Step 1: 膨胀所有 mask
    dilated_masks = []
    kernel = np.ones((3, 3), np.uint8)
    for m in masks:
        dilated = cv2.dilate(m.astype(np.uint8), kernel, iterations=iterations)
        dilated_masks.append(dilated)

    # Step 2: 绘制每个 mask 的边界（绿色）
    for idx, m in enumerate(dilated_masks):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)  # green for all

    # Step 3: 检测所有两两交集
    for i in range(len(dilated_masks)):
        for j in range(i + 1, len(dilated_masks)):
            overlap = cv2.bitwise_and(dilated_masks[i], dilated_masks[j])
            if np.sum(overlap) == 0:
                continue
            # 高亮交集区域（红色边框）
            overlap_contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, overlap_contours, -1, (255, 0, 0), 2)  # red = intersection
            cv2.putText(vis, f"{i} ∩ {j}", (10, 30 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Step 4: 保存结果图
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved expanded mask overlap visualization to: {save_path}")

def dilate_mask(mask, iterations=2):
    """
    对 mask 进行膨胀处理，增加边界接触机会。
    iterations 越大，膨胀越多。
    """
    kernel = np.ones((3, 3), np.uint8)  # 3x3 卷积核
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

def check_proximity_occlusion(masks, affordance_masks, sam_centers, depth_image,
                               class_ids, class_names, depth_threshold=10):
    """
    基于 affordance 区域检测靠近遮挡：
    如果其他物体遮住了当前物体的 affordance 区域，且在前面，则视为遮挡。
    返回遮挡对 (i 被 j 遮挡)
    """
    occlusion_pairs = []

    for i, aff_mask in enumerate(affordance_masks):
        center_i = sam_centers[i]
        depth_i = robust_center_depth(depth_image, int(center_i[0]), int(center_i[1]), win=5)
        if depth_i is None:
            continue

        for j, m_j in enumerate(masks):
            if i == j:
                continue

            overlap = cv2.bitwise_and(aff_mask.astype(np.uint8), m_j.astype(np.uint8))
            if np.sum(overlap) == 0:
                continue  # 没有靠近或覆盖

            # 从交叉区域采样深度（只取最浅的）
            ys, xs = np.where(overlap > 0)
            sampled_depths = depth_image[ys, xs]
            valid_depths = sampled_depths[(sampled_depths > 0) & (sampled_depths < 10000)]
            if valid_depths.size == 0:
                continue
            min_depth_j = np.min(valid_depths)

            if min_depth_j + depth_threshold < depth_i:
                name_i = class_names[class_ids[i]]
                name_j = class_names[class_ids[j]]
                print(f"[Proximity Occlusion] Object {i} ({name_i}) is occluded by {j} ({name_j}) on affordance area.")
                print(f" - depth_i: {depth_i}, min_depth_j: {min_depth_j}, Δ={depth_i - min_depth_j}")
                occlusion_pairs.append((i, j))

    return occlusion_pairs

if __name__ == "__main__":
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "test_imgs/img7"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    color_image = cv2.imread(img_path)
    data = np.load(npz_path)
    depth_image = data["depth"]
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
    mboxes, mclass_ids, mcenters, groups = merge_detections_by_depth_and_proximity(
        boxes=boxes,
        class_ids=class_ids,
        centers_xy=yolo_centers,
        depth_image=depth_image.astype(np.uint16),
        pixel_dist_thresh=60,
        depth_diff_thresh=20,
        same_class_only=True,
        win=3
    )
    masks, sam_centers, result_path = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=mboxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )
    
    affordance_masks = []
    image_with_aff = image_rgb.copy()
    for i, m in enumerate(masks):
        grasp_angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)
        mean = np.array(sam_centers[i], dtype=np.float32)
        aff_mask = compute_affordance_mask(
            mask=m,
            major_axis=major_axis,
            minor_axis=minor_axis,
            mean=mean,
            center_ratio=0.3,
            expand_pixels=20
        )
        affordance_masks.append(aff_mask)
        image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask, color=(0, 255, 0), alpha=0.4)
    cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))
    print("\n===== Occlusion Analysis =====")
    occlusion_result = check_occlusions(
        masks=masks,
        affordance_masks=affordance_masks,
        sam_centers=sam_centers,
        depth_image=depth_image,
        class_ids=mclass_ids,
        class_names=class_names,
        depth_threshold=20,
        aff_threshold=10,
        iterations=1
    )
    # 可访问：
    depth_occluded = occlusion_result["depth_occlusion"]
    proximity_occluded = occlusion_result["proximity_occlusion"]

    draw_mask_intersections_expanded(image_rgb, masks, iterations=1, save_path="expanded_mask_overlap.png")
