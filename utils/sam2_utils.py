# sam2_utils.py
from ultralytics import SAM
import numpy as np
import cv2
import os
from typing import List, Tuple

# from .rotate import extract_grasp_infos
# from .rotate import simple_quaternion_from_angle

def segment_with_sam2(image_rgb, boxes, sam_model_path, input_image_path):
    """
    Perform precise segmentation using Ultralytics SAM2 for YOLO-detected bounding boxes,
    and draw the center point for each segmented object.

    Args:
        image_rgb (np.ndarray): Input RGB image of shape (H, W, 3)
        boxes (list of tuple): YOLO bounding boxes as [(x1, y1, x2, y2), ...]
        sam_model_path (str): Path to the SAM2 weights, e.g., "sam2_b.pt"
        input_image_path (str): Path of the original image (used for saving output)

    Returns:
        out_masks (list of np.ndarray): List of binary masks for each object
        centers (list of tuple): List of center points (u, v) for each mask
        saved_path (str): Path to the saved output image with masks and center points drawn
    """
    model = SAM(sam_model_path)

    # Save temporary image to disk (required by Ultralytics SAM API)
    temp_input_path = input_image_path
    if isinstance(image_rgb, np.ndarray):
        cv2.imwrite(temp_input_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Convert bounding boxes to list of int
    bbox_list = [list(map(int, box)) for box in boxes]

    # Run SAM2 inference
    results = model(temp_input_path, bboxes=bbox_list)

    # Get image with masks and boxes drawn
    output_img = results[0].plot()
    output_img = output_img.copy()

    # Extract masks and center points
    out_masks = []
    centers = []
    for i, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy().astype(np.uint8)
        out_masks.append(mask_np)

        if np.any(mask_np):
            yx = np.argwhere(mask_np)
            cy, cx = yx.mean(axis=0).astype(int)
            centers.append((cx, cy))

            # Draw center point on output image
            cv2.circle(output_img, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(output_img, f"#{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            centers.append((0, 0))

    # Save output image
    save_path = input_image_path.replace(".png", "_sam2_masks_with_centers.png")
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    print(f"[Segmentation] SAM2 finished. {len(centers)} objects segmented.")
    # print(f"[Output] Saved mask visualization with centers to: {save_path}")
    return out_masks, centers, save_path


    """自动缩小 ROI 大小，避免整面背景 mask"""
    r = r0
    while r >= rmin:
        x1, y1 = max(u - r, 0), max(v - r, 0)
        x2, y2 = min(u + r, W - 1), min(v + r, H - 1)
        results = model(image_path, bboxes=[[x1, y1, x2, y2]])
        mask_list = []
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for m in results[0].masks.data:
                mask_list.append(m.cpu().numpy().astype(np.uint8))
        best = _pick_mask(mask_list, (u, v), image_shape=(H, W))
        if best is not None and best.sum() / (H * W) < 0.3:
            return best
        r = int(r / 1.5)
    return None

def shrink_mask(mask, all_masks, center, area_ratio_thresh, min_radius, max_radius):
    """
    如果mask面积过大，就返回一个与其他mask不重合的小圆形区域
    """
    H, W = mask.shape
    u, v = center
    area = int(mask.sum())
    area_ratio = area / (H * W)
    if area_ratio <= area_ratio_thresh:
        print("normal mask")
        return mask
        

    union_mask = np.zeros_like(mask, dtype=np.uint8)
    for m in all_masks:
        union_mask = cv2.bitwise_or(union_mask, m.astype(np.uint8))

    for r in range(min_radius, max_radius + 1, 2):
        Y, X = np.ogrid[:H, :W]
        circle = ((X - u) ** 2 + (Y - v) ** 2 <= r ** 2).astype(np.uint8)
        if np.sum(circle & union_mask) == 0:
            return circle
    return mask

def _pick_mask(mask_list, center, image_shape):
    u, v = center
    for m in mask_list:
        if m[v, u]:  # ensure point is in mask
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

        if prefer_points_prompt:
            try:
                results = model(input_image_path, points=[[(u, v)]], labels=[[1]])
            except Exception as e:
                print(f"[ERROR] 点提示调用失败：{e}")
                results = None

        if results is None:
            # fallback to bbox around point
            r = point_radius_px
            x1, y1 = max(u - r, 0), max(v - r, 0)
            x2, y2 = min(u + r, W - 1), min(v + r, H - 1)
            results = model(input_image_path, bboxes=[[x1, y1, x2, y2]])

        mask_list = []
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for m in results[0].masks.data:
                mask_list.append(m.cpu().numpy().astype(np.uint8))

        best_mask = _pick_mask(mask_list, (u, v), image_shape=(H, W))
        if best_mask is None:
            best_mask = np.zeros((H, W), dtype=np.uint8)

        # 如果 mask 面积太大，缩成小圆形
        best_mask = shrink_mask(
            mask=best_mask,
            all_masks=masks,
            center=(u, v),
            area_ratio_thresh=area_ratio_thresh,
            min_radius=30,
            max_radius=300
        )

        masks.append(best_mask)

        # 计算中心点
        if best_mask.any():
            ys, xs = np.nonzero(best_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx, cy = u, v
        centers.append((cx, cy))

        # 可视化
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 165, 255), 2)
        cv2.circle(overlay, (u, v), 6, (255, 255, 255), -1)
        cv2.circle(overlay, (u, v), 4, (0, 140, 255), -1)
        cv2.putText(overlay, f"#{i}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

    out_path = input_image_path.replace(".png", "_sam2_combined.png")
    cv2.imwrite(out_path, overlay)

    return masks, centers, out_path

'''
def segment_with_sam2_points(
    image_rgb,
    points_uv,
    sam_model_path,
    input_image_path,
    point_radius_px=32,
    prefer_points_prompt=True
):
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

        if prefer_points_prompt:
            try:
                results = model(input_image_path, points=[[(u, v)]], labels=[[1]])
            except Exception as e:
                print(f"[ERROR] 点提示调用失败：{e}")
                results = None

        if results is None:
            # fallback to bbox around point
            r = point_radius_px
            x1, y1 = max(u - r, 0), max(v - r, 0)
            x2, y2 = min(u + r, W - 1), min(v + r, H - 1)
            results = model(input_image_path, bboxes=[[x1, y1, x2, y2]])

        # 选最佳 mask
        mask_list = []
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for m in results[0].masks.data:
                mask_list.append(m.cpu().numpy().astype(np.uint8))

        best_mask = _pick_mask(mask_list, (u, v), image_shape=(H, W))

        # A/B逻辑：mask 太大 → 自动剪裁 or 重试
        if best_mask is not None:
            area_ratio = best_mask.sum() / (H * W)
            if area_ratio > 0.35:
                print(f"[WARN] mask {i} 太大（{area_ratio:.2%}），触发自动处理...")
                best_mask_retry = _shrink_roi_sam(model, input_image_path, u, v, H, W)
                if best_mask_retry is not None and best_mask_retry.sum() < best_mask.sum():
                    best_mask = best_mask_retry
                    print(f"[INFO] mask {i} 缩小后成功，面积 = {best_mask.sum()} px")
                else:
                    # 最后保险方案：裁成圆形
                    Y, X = np.ogrid[:H, :W]
                    disk = ((X - u) ** 2 + (Y - v) ** 2) <= (point_radius_px ** 2)
                    best_mask = disk.astype(np.uint8)
                    print(f"[INFO] mask {i} 裁成点邻域圆盘")

        if best_mask is None:
            best_mask = np.zeros((H, W), dtype=np.uint8)

        masks.append(best_mask)

        if best_mask.any():
            ys, xs = np.nonzero(best_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            cx, cy = u, v
        centers.append((cx, cy))

        # 可视化
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 165, 255), 2)
        cv2.circle(overlay, (u, v), 6, (255, 255, 255), -1)
        cv2.circle(overlay, (u, v), 4, (0, 140, 255), -1)
        cv2.putText(overlay, f"#{i}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

    # 保存图像
    out_path = input_image_path.replace(".png", "_sam2_point_mask_combined.png") \
                               .replace(".jpg", "_sam2_point_mask_combined.jpg")
    cv2.imwrite(out_path, overlay)
    return masks, centers, out_path
'''

if __name__ == "__main__":
    '''
    from yolo_utils import detect_fruits

    image_path = "imgs/img_from_camera.png"
    sam_model_path = "checkpoints/sam2_b.pt"
    yolo_model_path = "checkpoints/best_yolo.pt"

    # 使用你原来的 YOLO 检测函数
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(image_path, yolo_model_path)

    # 使用 Ultralytics SAM2 做精细分割
    masks, sam_centers, result_path = segment_with_sam2(image_rgb, boxes, sam_model_path, image_path)
    print("SAM2中心点：", sam_centers)
    
    # grasp_infos, output_img_path = extract_grasp_infos(
    # image_rgb=image_rgb,
    # masks=masks,
    # pixel_to_meter=0.001,
    # output_path="grasp_infos_visual.png"
    # )

    # # 输出结果
    # print(" 抓取信息如下：")
    # for info in grasp_infos:
    #     print(info)
    # for info in grasp_infos:
    #     angle = info["angle_deg"]
    #     quat = simple_quaternion_from_angle(angle)
    #     # print(f"物体 #{info['index']} 四元数姿态: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")

    # print(" 可视化图已保存至：", output_img_path)
    '''
    bgr = cv2.imread("imgs/3.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    points_uv = [(381, 78), (335, 400)]  # 举例

    masks, centers, vis = segment_with_sam2_points(
        image_rgb=rgb,
        points_uv=points_uv,
        sam_model_path="checkpoints/sam2_b.pt",
        input_image_path="imgs/3.png",
        point_radius_px=32,
        prefer_points_prompt=True
    )