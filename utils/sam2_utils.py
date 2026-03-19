"""
SAM2 utilities for precise object segmentation using bounding boxes or point prompts.
"""

from ultralytics import SAM
import numpy as np
import cv2
import os

def segment_with_sam2(image_rgb, boxes, sam_model_path, input_image_path):
    """Performs precise segmentation for YOLO-detected bounding boxes using SAM2."""
    model = SAM(sam_model_path)

    # Ensure input image is saved to disk for Ultralytics API compatibility
    if isinstance(image_rgb, np.ndarray):
        cv2.imwrite(input_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Convert bounding boxes and run inference
    bbox_list = [list(map(int, box)) for box in boxes]
    results = model(input_image_path, bboxes=bbox_list)

    output_img = results[0].plot().copy()
    out_masks, centers = [], []

    # Extract masks and compute geometric centers
    for i, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy().astype(np.uint8)
        out_masks.append(mask_np)

        if np.any(mask_np):
            yx = np.argwhere(mask_np)
            cy, cx = yx.mean(axis=0).astype(int)
            centers.append((cx, cy))
            cv2.circle(output_img, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(output_img, f"#{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            centers.append((0, 0))

    # Save visualization result
    save_path = input_image_path.replace(".png", "_sam2_masks_with_centers.png")
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    print(f"[INFO] SAM2 Segmentation finished: {len(centers)} objects processed.")
    return out_masks, centers, save_path

def shrink_mask(mask, all_masks, center, area_ratio_thresh, min_radius, max_radius):
    """Generates a small circular fallback mask if the original mask is too large or leaks to background."""
    H, W = mask.shape
    u, v = center
    area_ratio = int(mask.sum()) / (H * W)
    
    if area_ratio <= area_ratio_thresh:
        return mask

    # Combine existing masks to avoid overlap
    union_mask = np.zeros_like(mask, dtype=np.uint8)
    for m in all_masks:
        union_mask = cv2.bitwise_or(union_mask, m.astype(np.uint8))

    # Incrementally search for a valid non-overlapping circular region
    for r in range(min_radius, max_radius + 1, 2):
        Y, X = np.ogrid[:H, :W]
        circle = ((X - u) ** 2 + (Y - v) ** 2 <= r ** 2).astype(np.uint8)
        if np.sum(circle & union_mask) == 0:
            print(f"[DEBUG] Oversized mask detected. Falling back to circular ROI (r={r}).")
            return circle
            
    return mask

def _pick_mask(mask_list, center):
    """Selects the best mask from a list that contains the prompt point."""
    u, v = center
    for m in mask_list:
        if m[v, u]: 
            return m
    return None

def segment_with_sam2_points(
    image_rgb,
    points_uv,
    sam_model_path,
    input_image_path,
    point_radius_px=32,
    prefer_points_prompt=True,
    area_ratio_thresh=0.35
):
    """Segments objects using point prompts or local BBox fallbacks."""
    model = SAM(sam_model_path)
    
    if isinstance(image_rgb, np.ndarray):
        cv2.imwrite(input_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    img = cv2.imread(input_image_path)
    H, W = img.shape[:2]
    overlay = img.copy()
    masks, centers = [], []

    for i, (u, v) in enumerate(points_uv):
        u, v = int(np.clip(u, 0, W - 1)), int(np.clip(v, 0, H - 1))
        results = None

        # Attempt point-based prompt
        if prefer_points_prompt:
            try:
                results = model(input_image_path, points=[[(u, v)]], labels=[[1]])
            except Exception as e:
                print(f"[ERROR] Point prompt failed: {e}")

        # Fallback to a local bounding box if point prompt fails
        if results is None:
            r = point_radius_px
            x1, y1 = max(u - r, 0), max(v - r, 0)
            x2, y2 = min(u + r, W - 1), min(v + r, H - 1)
            results = model(input_image_path, bboxes=[[x1, y1, x2, y2]])

        mask_list = []
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for m in results[0].masks.data:
                mask_list.append(m.cpu().numpy().astype(np.uint8))

        # Select the most relevant mask and apply area constraints
        best_mask = _pick_mask(mask_list, (u, v))
        if best_mask is None:
            best_mask = np.zeros((H, W), dtype=np.uint8)

        best_mask = shrink_mask(
            mask=best_mask,
            all_masks=masks,
            center=(u, v),
            area_ratio_thresh=area_ratio_thresh,
            min_radius=30,
            max_radius=300
        )

        masks.append(best_mask)

        # Compute geometric center for the final mask
        if best_mask.any():
            ys, xs = np.nonzero(best_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx, cy = u, v
        centers.append((cx, cy))

        # Visualization: draw contours and index labels
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 165, 255), 2)
        cv2.circle(overlay, (u, v), 6, (255, 255, 255), -1)
        cv2.circle(overlay, (u, v), 4, (0, 140, 255), -1)
        cv2.putText(overlay, f"#{i}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

    out_path = input_image_path.replace(".png", "_sam2_combined.png")
    cv2.imwrite(out_path, overlay)

    return masks, centers, out_path