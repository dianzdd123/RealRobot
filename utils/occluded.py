import numpy as np
import cv2
import os
import pyrealsense2 as rs
from ultralytics import YOLO
from yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from sam2_utils import segment_with_sam2
from affordance_area import compute_affordance_mask, draw_affordance_on_image
from rotate import infer_grasp_angle_and_width

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
'''
def check_depth_occlusion(depth_image, mask, center, all_masks, self_mask_index,
                          class_ids, class_names, debug=False,
                          depth_diff_threshold=10):
    """
    判断当前物体 mask 是否被其他物体遮挡（深度遮挡判断）
    """
    h, w = depth_image.shape
    x, y = center
    if not (0 <= x < w and 0 <= y < h):
        return False, None

    center_depth = robust_center_depth(depth_image, x, y, win=5)
    if center_depth is None:
        return False, None

    # 提取当前物体边界点
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) == 0:
        return False, None

    boundary_points = contours[0][:, 0, :]  # (N, 2)

    for pt in boundary_points:
        px, py = pt
        if not (0 <= px < w and 0 <= py < h):
            continue

        # 检查该边界点是否与其他物体 mask 重叠
        for j, other_mask in enumerate(all_masks):
            if j == self_mask_index:
                continue

            if other_mask[py, px] > 0:  # 重叠了
                bd = robust_center_depth(depth_image, px, py, win=3)
                if bd is None:
                    continue
                depth_diff = center_depth - bd
                if depth_diff > depth_diff_threshold:
                    if debug:
                        name_i = class_names[class_ids[self_mask_index]]
                        name_j = class_names[class_ids[j]]
                        print(f"[Depth Occlusion] Object {self_mask_index} ({name_i}) is occluded by object {j} ({name_j})")
                        print(f" - Boundary point: ({px}, {py}), center depth: {center_depth}, boundary depth: {bd}, Δ={depth_diff:.1f}")
                    return True, ((x, y), (px, py))

    if debug:
        name_i = class_names[class_ids[self_mask_index]]
        print(f"[Depth Check] Object {self_mask_index} ({name_i}) → no depth occlusion found.")

    return False, None
'''
def check_occlusion(depth_image, mask, affordance_mask, center,
                    all_masks=None, all_affordance_masks=None,
                    depth_diff_threshold=10, debug=False,
                    class_ids=None, class_names=None, self_idx=None):
    """
    返回:
      ok: bool
      occluder_idx: int or None
      occ_type: 'depth' 或 'proximity' 或 None
      contact_pair: ((cx,cy),(px,py)) 或 None
    """
    x, y = center
    h, w = depth_image.shape
    if not (0 <= x < w and 0 <= y < h):
        return False, None, None, None

    center_depth = robust_center_depth(depth_image, x, y, win=5)
    if center_depth is None:
        return False, None, None, None

    # 1) 深度遮挡：你的边界比中心更近，且邻域属于别的物体
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        boundary_points = contours[0][:, 0, :]
        win = 2
        for pt in boundary_points:
            px, py = pt
            if not (0 <= px < w and 0 <= py < h):
                continue
            bd = robust_center_depth(depth_image, px, py, win=3)
            if bd is None:
                continue
            depth_diff = center_depth - bd
            if depth_diff > depth_diff_threshold:
                for j, other_mask in enumerate(all_masks):
                    if j == self_idx:
                        continue
                    hit = False
                    for dx in range(-win, win + 1):
                        for dy in range(-win, win + 1):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < w and 0 <= ny < h and other_mask[ny, nx] > 0:
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        if debug:
                            name_i = class_names[class_ids[self_idx]]
                            name_j = class_names[class_ids[j]]
                            print(f"[Depth Occlusion] {self_idx} ({name_i}) ← {j} ({name_j})")
                        return True, j, 'depth', ((x, y), (px, py))

    # 2) 靠近遮挡：affordance 区与他者 mask 有交叠，且对方不更远
    if all_masks is not None and all_affordance_masks is not None:
        aff_mask_bin = affordance_mask.astype(np.uint8)

        for j, other_mask in enumerate(all_masks):
            if j == self_idx:
                continue

            overlap = cv2.bitwise_and(aff_mask_bin, other_mask.astype(np.uint8))
            if np.count_nonzero(overlap) == 0:
                continue

            # 采样计算两侧近端深度
            aff_coords = np.column_stack(np.nonzero(aff_mask_bin))
            other_coords = np.column_stack(np.nonzero(other_mask))
            if aff_coords.size == 0 or other_coords.size == 0:
                continue

            step_aff = max(1, len(aff_coords)//800)
            step_oth = max(1, len(other_coords)//800)
            aff_depths = [robust_center_depth(depth_image, x, y, win=3) for y, x in aff_coords[::step_aff]]
            other_depths = [robust_center_depth(depth_image, x, y, win=3) for y, x in other_coords[::step_oth]]

            aff_depths = [d for d in aff_depths if d is not None]
            other_depths = [d for d in other_depths if d is not None]
            if not aff_depths or not other_depths:
                continue

            min_aff = np.percentile(aff_depths, 30)
            min_other = np.percentile(other_depths, 30)

            if min_other <= min_aff + 10:
                if debug:
                    name_i = class_names[class_ids[self_idx]]
                    name_j = class_names[class_ids[j]]
                    print(f"[Proximity Occlusion] {self_idx} ({name_i}) ← {j} ({name_j})")
                return True, j, 'proximity', None

    return False, None, None, None

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
    path_base = "test_imgs/img2"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"

    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names
    
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # pipeline.start(config)

    # align = rs.align(rs.stream.color)
    # # Warm up the camera
    # for _ in range(10):
    #     frames = pipeline.wait_for_frames()

    # # Capture one aligned RGB-D frame
    # frames = pipeline.wait_for_frames()

    # aligned_frames = align.process(frames)
    # color_frame = aligned_frames.get_color_frame()
    # depth_frame = aligned_frames.get_depth_frame()
    # if not color_frame or not depth_frame:
    #     print("[ERROR] Failed to capture frames.")
    #     exit(1)

    # # Convert frames to numpy arrays
    # color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    # # Save RGB and depth images
    # success = cv2.imwrite(img_path, color_image)
    # if not success:
    #     print("[ERROR] RGB图像保存失败！")

    # np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    # pipeline.stop()
    
    data = np.load(npz_path)
    color_image = data["color"]
    depth_image = data["depth"]

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
            center_ratio=0.3,
            expand_pixels=20
        )

        affordance_masks.append(aff_mask)
        # 绘制叠加
        image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask, color=(0, 255, 0), alpha=0.4)
    # 保存叠加可视化结果
    cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))

    # 遮挡检测
    occlusion_line_pairs = []
    for idx, mask in enumerate(masks):
        class_id = class_ids[idx]
        class_name = class_names.get(class_id, f"class_{class_id}")
        center = sam_centers[idx]

        is_occluded, contact_pair, _, _ = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=affordance_masks[idx],
            center=center,
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            debug=True,
            class_ids=class_ids,
            class_names=class_names,
            self_idx=idx
        )


    vis_img = draw_mask_boundaries(
        image_rgb,
        masks,
        ids=list(range(len(masks))),
        line_pairs=occlusion_line_pairs
    )
    cv2.imwrite("imgs/occluded.png", vis_img)