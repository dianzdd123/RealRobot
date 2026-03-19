import numpy as np
import cv2
import os
import json
import pyrealsense2 as rs
from ultralytics import YOLO
from .yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from .sam2_utils import segment_with_sam2
from .affordance_area import compute_affordance_mask, draw_affordance_on_image
from .rotate import infer_grasp_angle_and_width

# Configuration Constants
MIN_AFFORDANCE_PIXELS = 150  # Areas smaller than this are marked as non-graspable
DEPTH_DIFF_THRESHOLD = 10    # Threshold for depth-based occlusion (mm)

def robust_center_depth(depth_image, cx, cy, win=5):
    """Retrieves the median depth value from a local window for stability."""
    h, w = depth_image.shape
    cx = int(round(cx)); cy = int(round(cy))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    x0, x1 = max(cx - win, 0), min(cx + win + 1, w)
    y0, y1 = max(cy - win, 0), min(cy + win + 1, h)
    patch = depth_image[y0:y1, x0:x1].astype(np.uint16)
    vals = patch[patch > 0]
    return int(np.median(vals)) if vals.size > 0 else None

def check_occlusion(depth_image, mask, affordance_mask, center,
                    all_masks=None, all_affordance_masks=None,
                    depth_diff_threshold=10, debug=False,
                    class_ids=None, class_names=None, self_idx=None):
    """Detects both depth-edge and proximity-based occlusions for a target object."""
    x, y = center
    h, w = depth_image.shape
    if not (0 <= x < w and 0 <= y < h):
        return False, None, None, None

    center_depth = robust_center_depth(depth_image, x, y, win=5)
    if center_depth is None:
        return False, None, None, None

    # 1) Depth Occlusion: Check if boundary is significantly closer than center
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
                        if hit: break
                    if hit:
                        if debug:
                            name_i = class_names[class_ids[self_idx]]
                            name_j = class_names[class_ids[j]]
                            print(f"[DEBUG] Depth Occlusion: {self_idx}({name_i}) occluded by {j}({name_j})")
                        return True, j, 'depth', ((x, y), (px, py))

    # 2) Proximity Occlusion: Check if affordance region overlaps with closer objects
    if all_masks is not None and all_affordance_masks is not None:
        aff_mask_bin = affordance_mask.astype(np.uint8)

        for j, other_mask in enumerate(all_masks):
            if j == self_idx:
                continue

            overlap = cv2.bitwise_and(aff_mask_bin, other_mask.astype(np.uint8))
            if np.count_nonzero(overlap) == 0:
                continue

            # Sample depths to compare relative distances
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

            if min_other <= min_aff + 20:
                if debug:
                    name_i = class_names[class_ids[self_idx]]
                    name_j = class_names[class_ids[j]]
                    print(f"[DEBUG] Proximity Occlusion: {self_idx}({name_i}) blocked by {j}({name_j})")
                return True, j, 'proximity', None

    return False, None, None, None

def draw_mask_boundaries(image, masks, color=(0, 255, 0), thickness=2, ids=None, line_pairs=None):
    """Draws mask contours and label markers for occlusion visualization."""
    vis_image = image.copy()
    for idx, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(vis_image, contours, -1, color, thickness)
        if ids is not None:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                cv2.putText(vis_image, str(ids[idx]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if line_pairs:
        for pt1, pt2 in line_pairs:
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)
    return vis_image

def compute_grasp_difficulty(masks, affordance_masks):
    """Calculates graspability flags and difficulty scores based on affordance pixel ratio."""
    graspable_flags, grasp_difficulties = [], []
    for m, aff in zip(masks, affordance_masks):
        aff_pixels = np.count_nonzero(aff)
        total_pixels = np.count_nonzero(m)
        if aff_pixels < MIN_AFFORDANCE_PIXELS:
            graspable_flags.append(False)
            grasp_difficulties.append(1.0)
        else:
            graspable_flags.append(True)
            grasp_difficulties.append(1 - aff_pixels / total_pixels)
    return graspable_flags, grasp_difficulties

def save_occlusion_info(occlusion_info, class_ids, class_names, output_path):
    """Maps numerical occlusion relationships to human-readable labels and saves to JSON."""
    def id_to_name(target):
        if isinstance(target, int):
            cid = class_ids[target] if isinstance(class_ids, (list, tuple)) and 0 <= target < len(class_ids) else target
            if isinstance(class_ids, dict): cid = class_ids.get(target, target)
        else:
            cid = class_ids.get(target, target) if isinstance(class_ids, dict) else target

        if isinstance(class_names, dict): return str(class_names.get(cid, cid))
        if isinstance(class_names, (list, tuple)) and isinstance(cid, int) and 0 <= cid < len(class_names):
            return str(class_names[cid])
        return str(cid)

    def parse_relation(rel):
        if isinstance(rel, (tuple, list)) and len(rel) >= 2:
            return rel[0], str(rel[1])
        if isinstance(rel, dict):
            j = rel.get('by') or rel.get('j') or rel.get('idx')
            occ_type = rel.get('type') or rel.get('kind')
            if j is not None and occ_type is not None: return j, str(occ_type)
        return None

    occlusion_data = []
    for i, relations in occlusion_info.items():
        if relations is None: relations = []
        occluded_by_list = []
        for rel in relations:
            parsed = parse_relation(rel)
            if parsed:
                j, occ_type = parsed
                occluded_by_list.append({"by": id_to_name(j), "type": occ_type})
        occlusion_data.append({"object": id_to_name(i), "occluded_by": occluded_by_list})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(occlusion_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Occlusion information saved to: {output_path}")

if __name__ == "__main__":
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/img_from_camera"
    img_path, npz_path = f"{path_base}.png", f"{path_base}.npz"

    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    for _ in range(10): pipeline.wait_for_frames() # Warm up sensor

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame, depth_frame = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        print("[ERROR] Failed to capture frames from RealSense.")
        exit(1)

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    cv2.imwrite(img_path, color_image)
    np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    pipeline.stop()
    
    # Perception Pipeline
    image_rgb, boxes, yolo_centers, class_ids, _ = detect_fruits(img_path, yolo_model_path)
    mboxes, mclass_ids, mcenters, _ = merge_detections_by_depth_and_proximity(
        boxes=boxes, class_ids=class_ids, centers_xy=yolo_centers,
        depth_image=depth_image.astype(np.uint16), pixel_dist_thresh=60, depth_diff_thresh=20,
        same_class_only=True, win=3
    )
    
    masks, sam_centers, _ = segment_with_sam2(image_rgb=image_rgb, boxes=mboxes, sam_model_path=sam2_model_path, input_image_path=img_path)

    # Affordance and Occlusion Analysis
    image_with_aff = image_rgb.copy()
    affordance_masks = []
    for i, m in enumerate(masks):
        _, _, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)
        aff_mask = compute_affordance_mask(mask=m, major_axis=major_axis, minor_axis=minor_axis, mean=np.array(sam_centers[i], dtype=np.float32), center_ratio=0.3, expand_pixels=20)
        affordance_masks.append(aff_mask)
        image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask)

    cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))

    for idx, mask in enumerate(masks):
        check_occlusion(
            depth_image=depth_image, mask=mask, affordance_mask=affordance_masks[idx],
            center=sam_centers[idx], all_masks=masks, all_affordance_masks=affordance_masks,
            debug=True, class_ids=class_ids, class_names=class_names, self_idx=idx
        )