import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2


def segment_with_sam2(image_rgb, boxes, sam_path, sm2_input_path):

    """
    使用 SAM2 对 YOLO 检测框进行精细分割

    参数:
        image_rgb (np.ndarray): 原始 RGB 图像
        boxes (list of tuple): [(x1, y1, x2, y2), ...]

    返回:
        masks (list of np.ndarray): 每个目标的 mask (H, W)
        new_centers (list of tuple): 每个 mask 的中心点 (u, v)
    """
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to("cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    input_boxes = torch.tensor(boxes, device="cuda")

    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=input_boxes,
        multimask_output=False,
    )

    new_centers = []
    out_masks = []
    image_with_mask = image_rgb.copy()

    for i in range(len(masks)):
        mask_np = masks[i][0].cpu().numpy()
        out_masks.append(mask_np)

        if np.any(mask_np):
            yx = np.argwhere(mask_np)
            cy, cx = yx.mean(axis=0).astype(int)
            new_centers.append((cx, cy))

            # 将 mask 可视化叠加（例如绿色半透明）
            overlay = np.zeros_like(image_with_mask, dtype=np.uint8)
            overlay[mask_np] = (0, 255, 0)
            image_with_mask = cv2.addWeighted(image_with_mask, 1.0, overlay, 0.5, 0)
        else:
            new_centers.append((0, 0))  # fallback
    
    sm2_output_path = sm2_input_path.replace(".png", "_sam2_masks.png")
    # 保存带 mask 的图像
    image_bgr = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sm2_output_path, image_bgr)
    print(new_centers)
    print(f"✅ SAM2 分割图像已保存：{sm2_output_path}")
    return out_masks, new_centers, sm2_output_path
# ====== main 函数测试 ======
if __name__ == "__main__":
    from yolo_utils import detect_fruits
    testimg = "test.png"
    yolo_path = "yolo_train/train_data_split/runs/detect/train/weights/best.pt"
    # sam_path = "checkpoints/sam2.1_hiera_large.pt"
    sam_path = "checkpoints/sam_vit_h.pth"
    # Step 2: YOLO 检测
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(testimg,yolo_path)
    print("✅ YOLO 检测完成，共识别到", len(boxes), "个物体")
    print(yolo_centers)

    # Step 3: SAM2 精细分割并获取 mask 中心点（可替换 YOLO 中心点）
    masks, sam_centers,_ = segment_with_sam2(image_rgb, boxes, sam_path, yolo_output_path)
    print(sam_centers)