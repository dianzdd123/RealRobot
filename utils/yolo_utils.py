"""
YOLO utilities for object detection and post-processing (merging and depth analysis).
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

def robust_center_depth(depth_image, cx, cy, win=5):
    """Retrieves the median depth around a point, ignoring zero (hole) values."""
    h, w = depth_image.shape
    cx = int(round(cx)); cy = int(round(cy))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    x0, x1 = max(cx - win, 0), min(cx + win + 1, w)
    y0, y1 = max(cy - win, 0), min(cy + win + 1, h)
    patch = depth_image[y0:y1, x0:x1].astype(np.uint16)
    vals = patch[patch > 0]
    return int(np.median(vals)) if vals.size > 0 else None

def depths_from_centers(depth_image, centers_xy, win=5):
    """Batch processes robust depth extraction for a list of center coordinates."""
    return [robust_center_depth(depth_image, cx, cy, win=win) for (cx, cy) in centers_xy]

def merge_detections_by_depth_and_proximity(
    boxes, class_ids, centers_xy, depth_image,
    pixel_dist_thresh=100, depth_diff_thresh=20, same_class_only=True, win=3
):
    """Merges redundant detections based on spatial proximity and depth similarity."""
    N = len(boxes)
    used = [False] * N
    groups = []

    # Pre-fetch robust depths for all centers
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

            # Merge criteria: class identity, depth validity, and spatial/depth thresholds
            if same_class_only and (ci != cj):
                continue
            if di is None or dj is None:
                continue

            dist = np.hypot(xi - xj, yi - yj)
            if dist <= pixel_dist_thresh and abs(di - dj) <= depth_diff_thresh:
                group.append(j); used[j] = True

        groups.append(group)

    # Reconstruct merged bounding boxes and centers
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

        # Merge boxes by taking the global min/max coordinates
        merged_boxes.append([min(xs1), min(ys1), max(xs2), max(ys2)])
        
        # Determine class ID using majority vote
        vals, counts = np.unique(cls_list, return_counts=True)
        merged_class_ids.append(int(vals[np.argmax(counts)]))
        
        # Calculate new center as the mean of grouped centers
        merged_centers.append((int(np.mean(cx_list)), int(np.mean(cy_list))))

    return merged_boxes, merged_class_ids, merged_centers, groups

def detect_fruits(input_path, yolo_path):
    """Performs object detection using YOLO and saves boxed visualization images."""
    yolo_model = YOLO(yolo_path)
    results = yolo_model(input_path, conf=0.3, iou=0.5)

    image_bgr = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    boxes, centers, class_ids = [], [], []
    names = yolo_model.names

    for box, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = map(int, box)
        boxes.append((x1, y1, x2, y2))
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        class_ids.append(cls)

        # Draw labels and bounding boxes for debugging
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    yolo_output_path = input_path.replace(".png", "_boxed.png")
    cv2.imwrite(yolo_output_path, image_bgr)
    
    print(f"[INFO] YOLO detection finished. Found {len(boxes)} objects.")
    return image_rgb, boxes, centers, class_ids, yolo_output_path