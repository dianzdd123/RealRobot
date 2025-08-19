import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
# from yolo_utils import detect_fruits
# from sam2_utils import segment_with_sam2

def compute_gripper_for_real(width_m, gripper_min=0, gripper_max=850, width_min=0.01, width_max=0.08, safety_margin=0.005):
    """
    根据物体宽度（米）计算夹爪张开/闭合值（0-850）

    参数：
        width_m (float): 物体宽度（单位：米）
        gripper_min (int): 夹爪完全闭合的值（默认 0）
        gripper_max (int): 夹爪完全张开的值（默认 850）
        width_min (float): 能抓的最小物体宽度（米）
        width_max (float): 能抓的最大物体宽度（米）
        safety_margin (float): 开口时额外增加的宽度（米）

    返回：
        open_val (int): 张开值（略宽）
        close_val (int): 闭合值（刚好夹住）
    """
    width_clamped = np.clip(width_m, width_min, width_max)

    gripper_range = gripper_max - gripper_min

    # 线性比例映射宽度到夹爪值
    ratio = (width_clamped - width_min) / (width_max - width_min)
    close_val = gripper_min + ratio * gripper_range

    # 安全张开略大一点（按比例扩展）
    open_width = width_clamped + safety_margin
    open_width = np.clip(open_width, width_min, width_max)
    open_ratio = (open_width - width_min) / (width_max - width_min)
    open_val = gripper_min + open_ratio * gripper_range

    return int(round(open_val)), int(round(close_val))

def compute_gripper_for_sim(w_obj_m, gripper_min=0.0, gripper_max=0.85, safety_margin=0.005, max_obj_width=0.08):
    """
    Compute gripper open and close values based on object width.

    Parameters:
        w_obj_m (float): Object width in meters.
        gripper_min (float): Gripper fully open position (usually 0.0).
        gripper_max (float): Gripper fully closed position (e.g., 0.85).
        safety_margin (float): Added/subtracted safety margin in meters.
        max_obj_width (float): Max object width gripper can handle.

    Returns:
        open_value (float): Gripper value to open before grasping.
        close_value (float): Gripper value to close for grasping.
    """
    # Clamp width
    w = np.clip(w_obj_m, 0.01, max_obj_width)

    # Total range = max - min
    gripper_range = gripper_max - gripper_min

    # Mapping: 0.0 width => 0.0 (fully open), max_obj_width => 0.85 (fully closed)
    ratio = w / max_obj_width
    base_value = gripper_min + ratio * gripper_range
    # Open: slightly wider
    open_value = np.clip(base_value + safety_margin * gripper_range / max_obj_width, gripper_min, gripper_max)
    # Close: slightly tighter
    close_value = np.clip(base_value - safety_margin * gripper_range / max_obj_width, gripper_min, gripper_max)

    return open_value, close_value

def infer_grasp_angle_and_width(mask, draw_on=None, object_index=0, pixel_to_meter=0.0001):
    """
    计算物体主轴方向（PCA）用于夹爪朝向，并获取副轴方向上的宽度。

    参数:
        mask (np.ndarray): 单个物体的二值掩膜图像 (H, W)，uint8
        draw_on (np.ndarray): 可选，绘制箭头的背景图像（RGB）
        object_index (int): 当前物体编号，用于标注
        pixel_to_meter (float): 可选，用于将像素宽度转换为真实长度

    返回:
        grasp_angle (float): 抓取时绕Z轴旋转的角度（单位：度）
        width (float): 物体在副轴方向上的宽度（像素或米）
        output_img (np.ndarray): 绘制主轴箭头后的图像（RGB）
    """
    ys, xs = np.nonzero(mask)
    coords = np.stack((xs, ys), axis=1)

    if len(coords) < 10:
        print("⚠️ mask 太小，跳过")
        return 0, 0, draw_on if draw_on is not None else np.zeros((*mask.shape, 3), dtype=np.uint8)

    # 1. 中心化坐标
    mean = np.mean(coords, axis=0)
    centered = coords - mean

    # 2. PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]  # 主轴（最长方向）
    minor_axis = eigvecs[:, np.argmin(eigvals)]  # 副轴（宽度方向）

    # 3. 计算夹爪旋转角度（绕 Z 轴）
    angle_rad = np.arctan2(minor_axis[1], minor_axis[0])
    grasp_angle = -np.degrees(angle_rad)
    
    # 4. 计算副轴方向上的投影长度（宽度）
    projections = centered @ minor_axis
    width_pixels = projections.max() - projections.min()
    width = width_pixels * pixel_to_meter if pixel_to_meter else width_pixels

    # 5. 绘图准备
    output_img = draw_on.copy() if draw_on is not None else cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    center_int = tuple(np.round(mean).astype(int))

    for vec, color, label in zip([major_axis, minor_axis], [(0, 255, 0), (0, 0, 255)], ["major", "minor"]):
        tip = (mean + vec * 50).astype(int)
        cv2.arrowedLine(output_img, center_int, tuple(tip), color=color, thickness=2, tipLength=0.2)
        cv2.putText(output_img, f"{label}_{object_index}", tuple(tip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return grasp_angle, width, output_img, major_axis, minor_axis

def extract_grasp_infos(image_rgb, masks, pixel_to_meter=0.0001, output_path="imgs/grasp_infos.png"):
    """
    提取每个 SAM2 mask 的抓取信息（中心点、主轴角度、夹爪值），不依赖类别信息。

    参数:
        image_rgb (np.ndarray): 原始 RGB 图像
        masks (List[np.ndarray]): 每个物体的 mask（二维 uint8 图）
        pixel_to_meter (float): 每像素代表的真实长度（单位：米），可选
        output_path (str): 输出可视化图像保存路径

    返回:
        grasp_infos (List[dict]): 每个物体的抓取信息（中心点、角度、夹爪值）
        output_path (str): 保存图像的路径
    """
    draw_img = image_rgb.copy()
    grasp_infos = []

    for i, mask in enumerate(masks):
        angle, width, draw_img, _, _, = infer_grasp_angle_and_width(
            mask=mask,
            draw_on=draw_img,
            object_index=i,
            pixel_to_meter=pixel_to_meter
        )
        # 计算 mask 的中心点（从 mask 本身的像素坐标平均）
        cy, cx = np.mean(np.argwhere(mask), axis=0).astype(int)

        grasp_infos.append({
            "index": i,
            "center": (cx, cy),
            "angle": round(angle, 2),
            "width": round(width, 2),
        })

        # 可视化中心点和编号
        cv2.circle(draw_img, (cx, cy), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.putText(draw_img, f"#{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # 保存最终图像
    draw_img_bgr = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, draw_img_bgr)

    return grasp_infos, output_path
 
def simple_quaternion_from_angle(angle_deg):

    # 将角度映射到 sin 曲线，角度范围 [-180, 180] → y 分量 [-1, 1]
    y = np.sin(np.radians(angle_deg))  # 角度转弧度

    # 构造非标准四元数结构（x, y, z, w）
    quat = np.array([0.0, y, 1.0, 0.0])

    # 归一化为单位四元数
    quat = quat / np.linalg.norm(quat)

    return np.round(quat, 4)  # 保留4位小数

def compute_affordance_mask(mask, major_axis, minor_axis, mean, center_ratio=0.4, expand_pixels=15):
    """
    生成可供性区域掩码：在主轴中心段截取，沿副轴方向扩展固定像素。

    参数:
        mask: 原始二值掩膜 (H, W)
        major_axis: 主轴向量（单位长度）
        minor_axis: 副轴向量（单位长度）
        mean: 中心坐标 (x, y)
        center_ratio: 主轴方向中段比例（如0.4为中间40%）
        expand_pixels: 沿副轴方向扩展的像素数
    返回:
        affordance_mask: 新掩码区域，uint8类型
    """

    h, w = mask.shape
    affordance_mask = np.zeros_like(mask, dtype=np.uint8)

    # Step 1: 主轴方向上截取中间区域
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)

    # 以 mean 为中心，投影到主轴方向
    relative = coords - mean  # N x 2
    proj_major = np.dot(relative, major_axis)  # N
    proj_minor = np.dot(relative, minor_axis)  # N

    # 计算主轴投影范围，截取中心比例区域
    proj_major_min = np.min(proj_major)
    proj_major_max = np.max(proj_major)
    proj_center = (proj_major_min + proj_major_max) / 2
    half_range = (proj_major_max - proj_major_min) * center_ratio / 2
    proj_mask = (proj_major >= proj_center - half_range) & (proj_major <= proj_center + half_range)

    selected_coords = coords[proj_mask]
    selected_proj_major = proj_major[proj_mask]
    selected_proj_minor = proj_minor[proj_mask]

    # Step 2: 沿副轴方向扩展
    expanded_coords = []

    for i in range(len(selected_coords)):
        for delta in [-expand_pixels, expand_pixels]:
            offset = minor_axis * delta
            new_point = selected_coords[i] + offset
            x, y = int(round(new_point[0])), int(round(new_point[1]))
            if 0 <= x < w and 0 <= y < h:
                expanded_coords.append((x, y))

    # 原始点也加入
    for pt in selected_coords:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            expanded_coords.append((x, y))

    # 绘制 mask
    for x, y in expanded_coords:
        affordance_mask[y, x] = 255

    # 可选：做膨胀以连通区域（例如5×5核）
    kernel = np.ones((5, 5), np.uint8)
    affordance_mask = cv2.dilate(affordance_mask, kernel, iterations=1)

    return affordance_mask

def draw_affordance_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


# if __name__ == "__main__":
#     yolo_model_path = "checkpoints/best_yolo.pt"
#     sam2_model_path = "checkpoints/sam2_b.pt"
#     path_base = "test_imgs/img4"
#     img_path = f"{path_base}.png"
#     npz_path = f"{path_base}.npz"
#     yolo_model = YOLO(yolo_model_path)
#     class_names = yolo_model.names

#     # 从 PNG 读取彩色图
#     color_image = cv2.imread(img_path)
#     image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)

#     # SAM 分割
#     masks, sam_centers, result_path = segment_with_sam2(
#         image_rgb=image_rgb,
#         boxes=boxes,
#         sam_model_path=sam2_model_path,
#         input_image_path=img_path
#     )

#     # 假设像素到米的比例（可根据相机标定得到）
#     pixel_to_meter = 0.001

#     # 1) 提取抓取信息
#     grasp_infos, output_path = extract_grasp_infos(
#         image_rgb=image_rgb,
#         masks=masks,
#         pixel_to_meter=pixel_to_meter,
#         output_path="output_grasp_vis.png"
#     )

#     image_with_aff = image_rgb.copy()
#     affordance_masks = []

#     for i, m in enumerate(masks):
#         aff_mask = compute_affordance_mask(m, center_ratio=0.3, width_expand_ratio=1.5)
#         affordance_masks.append(aff_mask)
#         image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask, color=(0, 255, 0), alpha=0.4)
#         print(f"物体 #{i} 的 affordance 区域已生成")

#     # 保存叠加可视化结果
#     cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))
#     print("✅ 叠加了 affordance 区域的图像已保存至 output_affordance_overlay.png")

#     # 3) 打印抓取结果
#     for info in grasp_infos:
#         print(f"物体 #{info['index']}:")
#         print(f"  中心点: {info['center']}")
#         print(f"  抓取角度: {info['angle']}°")
#         print(f"  宽度: {info['width']} m")

#         open_val, close_val = compute_gripper_for_real(info['width'])
#         print(f"  Gripper张开值: {open_val:.3f}, 闭合值: {close_val:.3f}")

#     print(f"抓取可视化图已保存至: {output_path}")