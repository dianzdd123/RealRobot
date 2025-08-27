import numpy as np
import cv2

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

def compute_affordance_striped_fullwidth(
    idx,
    masks,
    depth_image,
    class_ids,
    class_names,
    major_axes,
    minor_axes,
    prox_dilate_px=12,
    depth_dilate_px=6,
    depth_margin_mm=20,
    sample_step=2,
    local_win=3,
    stripe_bin_px=1
):
    my_mask = (masks[idx] > 0).astype(np.uint8)
    H, W = depth_image.shape
    if np.count_nonzero(my_mask) == 0:
        return np.zeros_like(my_mask), {
            "idx": idx, "class": class_names[class_ids[idx]],
            "obj_area": 0, "removed_stripes": 0,
            "afford_area": 0, "afford_ratio": 0.0, "note": "empty mask"
        }

    # === 1) 获取主轴 / 副轴向量 ===
    u_major = major_axes[idx]      # [ux, uy]
    u_minor = minor_axes[idx]      # [vx, vy]
    ux, uy = float(u_major[0]), float(u_major[1])
    vx, vy = float(u_minor[0]), float(u_minor[1])

    # === 2) 计算危险区（靠近 + 深度更靠前） ===
    kernel_prox  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prox_dilate_px*2+1,  prox_dilate_px*2+1))
    kernel_depth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (depth_dilate_px*2+1, depth_dilate_px*2+1))

    danger_prox = np.zeros_like(my_mask, dtype=np.uint8)
    others_union_depthband = np.zeros_like(my_mask, dtype=np.uint8)
    for j in range(len(masks)):
        if j == idx: continue
        other = (masks[j] > 0).astype(np.uint8)
        danger_prox = cv2.bitwise_or(danger_prox, cv2.dilate(other, kernel_prox))
        others_union_depthband = cv2.bitwise_or(others_union_depthband, cv2.dilate(other, kernel_depth))

    depth_forbid = np.zeros_like(my_mask, dtype=np.uint8)
    my_edge = cv2.morphologyEx(my_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    ys, xs = np.nonzero(my_edge)
    for k in range(0, len(xs), sample_step):
        x, y = xs[k], ys[k]
        if others_union_depthband[y, x] == 0:
            continue
        d_me = robust_center_depth(depth_image, x, y, win=local_win)
        if d_me is None: continue
        d_oth = robust_center_depth(depth_image, x, y, win=local_win)
        if d_oth is None: continue
        if d_oth + depth_margin_mm <= d_me:
            depth_forbid[y, x] = 1
    if np.count_nonzero(depth_forbid) > 0:
        depth_forbid = cv2.dilate(depth_forbid, kernel_depth)

    danger = cv2.bitwise_or(danger_prox, depth_forbid)
    danger = cv2.bitwise_and(danger, my_mask)

    # === 3) 投影到主轴，做条带判断 ===
    ys_m, xs_m = np.nonzero(my_mask)
    cx, cy = np.mean(xs_m), np.mean(ys_m)
    dx = xs_m - cx
    dy = ys_m - cy
    proj_major = dx * ux + dy * uy
    bins_m = np.round(proj_major / stripe_bin_px).astype(np.int32)

    ys_d, xs_d = np.nonzero(danger)
    bins_danger = set()
    if len(xs_d) > 0:
        dx_d = xs_d - cx
        dy_d = ys_d - cy
        proj_major_d = dx_d * ux + dy_d * uy
        bins_danger = set(np.round(proj_major_d / stripe_bin_px).astype(np.int32).tolist())

    keep = np.ones(len(xs_m), dtype=bool)
    if bins_danger:
        keep &= ~np.isin(bins_m, list(bins_danger))

    afford_mask = np.zeros_like(my_mask)
    afford_mask[ys_m[keep], xs_m[keep]] = 1

    # === 4) 最大连通域 ===
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(afford_mask, connectivity=8)
    final = np.zeros_like(afford_mask)
    removed_stripes = len(bins_danger)
    if nb > 1:
        largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final[labels == largest_id] = 1

    info = {
        "idx": idx,
        "class": class_names[class_ids[idx]],
        "obj_area": int(np.count_nonzero(my_mask)),
        "removed_stripes": int(removed_stripes),
        "afford_area": int(np.count_nonzero(final)),
    }
    info["afford_ratio"] = round(info["afford_area"]/max(1, info["obj_area"]), 4)
    return final, info

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

def visualize_graspables_fullmask(
    image_rgb,
    masks,
    graspables,
    class_ids,
    class_names,
    out_path,
    major_axes=None,
    minor_axes=None,
    centers=None
):
    vis = image_rgb.copy()

    # === 1. 绘制原始物体mask，淡绿色 ===
    for i, m in enumerate(masks):
        overlay = vis.copy()
        ys, xs = np.nonzero(m)
        overlay[ys, xs] = (overlay[ys, xs] * 0.5 + np.array((0, 255, 0)) * 0.5).astype(np.uint8)
        vis = overlay

    # === 2. 绘制可抓取区域，蓝色高亮 ===
    for i, g in enumerate(graspables):
        overlay = vis.copy()
        ys, xs = np.nonzero(g)
        overlay[ys, xs] = (overlay[ys, xs] * 0.3 + np.array((0, 0, 255)) * 0.7).astype(np.uint8)
        vis = overlay

        # === 3. 绘制物体标签 ===
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{i}:{class_names[class_ids[i]]}"
                cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
                cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # === 4. 绘制主轴(绿色)和副轴(红色) ===
    if major_axes is not None and minor_axes is not None and centers is not None:
        for i, center in enumerate(centers):
            cx, cy = int(center[0]), int(center[1])
            tip_major = (int(cx + major_axes[i][0] * 50), int(cy + major_axes[i][1] * 50))
            tip_minor = (int(cx + minor_axes[i][0] * 50), int(cy + minor_axes[i][1] * 50))

            # 主轴 - 绿色箭头
            cv2.arrowedLine(vis, (cx, cy), tip_major, (0, 255, 0), 2, tipLength=0.2)
            # 副轴 - 红色箭头
            cv2.arrowedLine(vis, (cx, cy), tip_minor, (0, 0, 255), 2, tipLength=0.2)

    # === 5. 保存结果 ===
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))