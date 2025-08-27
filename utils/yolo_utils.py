# yolo_utils.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

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
    pixel_dist_thresh=100, depth_diff_thresh=20, same_class_only=True, win=3
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

def detect_fruits(input_path, yolo_path):
    yolo_model = YOLO(yolo_path)
    results = yolo_model(input_path, conf=0.3, iou=0.5)  # 这里直接调高 iou 阈值，减少重复框

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

        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    yolo_output_path = input_path.replace(".png", "_boxed.png")
    cv2.imwrite(yolo_output_path, image_bgr)
    return image_rgb, boxes, centers, class_ids, yolo_output_path

# ====== main 函数测试 ======
# if __name__ == "__main__":
#     yolo_model_path = "checkpoints/best_yolo.pt"
#     path_base = "test_imgs/img3"
#     img_path = f"{path_base}.png"
#     npz_path = f"{path_base}.npz"

#     if not os.path.exists(img_path) or not os.path.exists(npz_path):
#         raise FileNotFoundError("缺少图像或深度文件，请检查 img_path/npz_path")

#     yolo_model = YOLO(yolo_model_path)
#     class_names = yolo_model.names  # dict: id->name

#     color_image = cv2.imread(img_path)
#     data = np.load(npz_path)
#     depth_image = data["depth"]

#     image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)

#     # 合并重复框
#     mboxes, mclass_ids, mcenters, groups = merge_detections_by_depth_and_proximity(
#         boxes=boxes,
#         class_ids=class_ids,
#         centers_xy=yolo_centers,
#         depth_image=depth_image.astype(np.uint16),
#         pixel_dist_thresh=100,
#         depth_diff_thresh=20,
#         same_class_only=True,
#         win=3
#     )