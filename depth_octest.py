import os
import cv2
import json
import numpy as np
from utils.rotate import infer_grasp_angle_and_width
from utils.affordance_area import visualize_graspables_fullmask, compute_affordance_striped_fullwidth, compute_affordance_mask
from utils.occluded import check_occlusion

from utils.sam2_utils import segment_with_sam2
from utils.yolo_utils import detect_fruits
from ultralytics import YOLO

def recompute_grasp_affordance(image_rgb, depth_image, masks, sam_centers, class_ids, class_names, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # === 1. 初始 affordance 区域计算 ===
    affordance_masks, grasp_angles, widths, major_axes, minor_axes = [], [], [], [], []
    for i, m in enumerate(masks):
        angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)
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
        grasp_angles.append(angle)
        widths.append(width)
        major_axes.append(major_axis)
        minor_axes.append(minor_axis)

    # === 2. 检测互相遮挡 ===
    mutual_pairs = set()
    for idx in range(len(masks)):
        ok, j, _, _ = check_occlusion(
            depth_image=depth_image,
            mask=masks[idx],
            affordance_mask=affordance_masks[idx],
            center=sam_centers[idx],
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            class_ids=class_ids,
            class_names=class_names,
            self_idx=idx
        )
        if ok and j is not None:
            # 再检查反向遮挡
            ok2, _, _, _ = check_occlusion(
                depth_image=depth_image,
                mask=masks[j],
                affordance_mask=affordance_masks[j],
                center=sam_centers[j],
                all_masks=masks,
                all_affordance_masks=affordance_masks,
                class_ids=class_ids,
                class_names=class_names,
                self_idx=j
            )
            if ok2:
                mutual_pairs.add(tuple(sorted((idx, j))))

    # === 3. 针对互相遮挡物体重算 affordance ===
    for idx1, idx2 in mutual_pairs:
        for idx in [idx1, idx2]:
            new_mask, _ = compute_affordance_striped_fullwidth(
                idx=idx,
                masks=masks,
                depth_image=depth_image.astype(np.uint16),
                class_ids=class_ids,
                class_names=class_names,
                major_axes=major_axes,
                minor_axes=minor_axes,
                prox_dilate_px=12,
                depth_dilate_px=6,
                depth_margin_mm=20,
                sample_step=2,
                local_win=3,
                stripe_bin_px=1
            )
            # 更新该物体的抓取区域
            affordance_masks[idx] = new_mask
            # 用新的区域重新提取特征
            angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(
                new_mask, draw_on=image_rgb, object_index=idx
            )
            grasp_angles[idx] = angle
            widths[idx] = width
            major_axes[idx] = major_axis
            minor_axes[idx] = minor_axis

    # === 4. 可视化最终结果 ===
    out_path = os.path.join(output_dir, "final_affordance.png")
    new_centers = visualize_graspables_fullmask(
        image_rgb=image_rgb,
        masks=masks,
        graspables=affordance_masks,
        class_ids=class_ids,
        class_names=class_names,
        out_path="results/final_affordance.png",
        major_axes=major_axes,
        minor_axes=minor_axes
    )
    print(f"[INFO] 可视化结果已保存：{out_path}")

    # === 5. 导出 JSON ===
    results = []
    for i in range(len(masks)):
        results.append({
            "id": int(i),
            "class": class_names[class_ids[i]],
            "center": new_centers,
            "angle": float(grasp_angles[i]),
            "width": float(widths[i]),
            "major_axis": major_axes[i].tolist(),
            "minor_axis": minor_axes[i].tolist()
        })

    json_path = os.path.join(output_dir, "grasp_info.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] 抓取信息已保存：{json_path}")

    return affordance_masks, grasp_angles, widths, major_axes, minor_axes


def main():
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    # path_base = "imgs/img_from_camera"
    path_base = "test_imgs/img5"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"

    if not os.path.exists(img_path) or not os.path.exists(npz_path):
        raise FileNotFoundError("缺少图像或深度文件，请检查 img_path/npz_path")

    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names  # dict: id->name

    color_image = cv2.imread(img_path)
    data = np.load(npz_path)
    depth_image = data["depth"]

    image_rgb, boxes, _, class_ids, _ = detect_fruits(img_path, yolo_model_path)

    # === 3. SAM2分割 ===
    masks, sam_centers, _ = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=boxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )

    # === 4. 遮挡检测 + 抓取区域计算 ===
    affordance_masks, grasp_angles, widths, major_axes, minor_axes = recompute_grasp_affordance(
        image_rgb=image_rgb,
        depth_image=depth_image,
        masks=masks,
        sam_centers=sam_centers,
        class_ids=class_ids,
        class_names=class_names,
        output_dir="results"
    )

    print("[INFO] 抓取区域和抓取信息计算完成")
    print(f"[INFO] 可抓取区域可视化图: results/final_affordance.png")
    print(f"[INFO] 抓取信息 JSON: results/grasp_info.json")


if __name__ == "__main__":
    main()