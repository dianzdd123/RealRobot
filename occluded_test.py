import cv2
import numpy as np
from utils.rotate import infer_grasp_angle_and_width, compute_affordance_mask

# ===================== 通用方案所需基础工具 =====================
def union_masks_by_idx(masks, idxs):
    """把一组 mask 合并成一个二值图"""
    if not idxs: return None
    H, W = masks[0].shape
    u = np.zeros((H, W), np.uint8)
    for k in idxs: u |= (masks[k] > 0).astype(np.uint8)
    return u

def keep_lcc(mask_u8):
    """保留 mask 中的最大连通域（避免碎片影响）"""
    if mask_u8 is None or mask_u8.sum() == 0: return np.zeros_like(mask_u8)
    num, lab = cv2.connectedComponents((mask_u8 > 0).astype(np.uint8))
    if num <= 1: return mask_u8
    best, best_id = 0, 0
    for i in range(1, num):
        area = int((lab == i).sum())
        if area > best: best, best_id = area, i
    return ((lab == best_id).astype(np.uint8)) * 255

# 用你的 PCA 函数包装成“从 mask 得姿态”
def pose_from_mask(mask_u8, pixel_to_meter=None):
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return None
    mean = (float(xs.mean()), float(ys.mean()))
    # 要像素宽度，传 pixel_to_meter=None
    yaw, width_px, _, major, minor = infer_grasp_angle_and_width(mask_u8, draw_on=None, object_index=0, pixel_to_meter=(None if pixel_to_meter is None else pixel_to_meter))
    return dict(center=mean, major=major, minor=minor, yaw=yaw, width_px=float(width_px))

# ===================== 类型化动作：切阴影 / 切重叠 / 削带 =====================
def cut_shadow_for_top(j_mask, lower_union, expand_px=15, safety_px=3):
    """处理混合遮挡：切掉上层 j 被下层入侵的阴影区域"""
    if lower_union is None: return j_mask.copy()
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*expand_px+1, 2*expand_px+1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*safety_px+1, 2*safety_px+1))
    shadow = cv2.dilate((lower_union > 0).astype(np.uint8), k1, 1)
    shadow = cv2.dilate(shadow, k2, 1)
    cut = cv2.bitwise_and((j_mask > 0).astype(np.uint8), cv2.bitwise_not(shadow))
    return keep_lcc(cut)

def cut_overlap_for_mutual(me_mask, neigh_union, expand_px=15, safety_px=3):
    """处理互相靠近遮挡：切掉自己与邻居的重叠区域"""
    if neigh_union is None: return me_mask.copy()
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*expand_px+1, 2*expand_px+1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*safety_px+1, 2*safety_px+1))
    me_dil   = cv2.dilate((me_mask > 0).astype(np.uint8), k1, 1)
    neigh_dil= cv2.dilate((neigh_union > 0).astype(np.uint8), k1, 1)
    overlap  = cv2.bitwise_and(me_dil, neigh_dil)
    if overlap.sum() == 0: return me_mask.copy()
    band = cv2.dilate(overlap, k2, 1)
    cut = cv2.bitwise_and((me_mask > 0).astype(np.uint8), cv2.bitwise_not(band))
    return keep_lcc(cut)

def carve_victim_side(victim_mask, victim_center_xy, minor_axis, clearance_px):
    """处理单向靠近遮挡：沿副轴方向切掉入侵侧的一条带（近侧）"""
    obj = (victim_mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(obj)
    if xs.size == 0: return victim_mask.copy()
    pts = np.stack([xs, ys], 1).astype(np.float32)
    rel = pts - np.array(victim_center_xy, np.float32)
    proj = rel @ minor_axis.astype(np.float32)
    keep = np.abs(proj) > float(clearance_px)
    cut = np.zeros_like(obj)
    kept = pts[keep].astype(np.int32)
    if kept.size: cut[kept[:,1], kept[:,0]] = 255
    return keep_lcc(cut)

# ===================== 主入口：生成安全抓取区域 =====================
def make_safe_region_for_object(
    idx, masks, prox, above,
    compute_affordance_mask,             # 直接传入你已有的函数
    expand_px=15, safety_px=3, min_area_px=200,
    orig_pose=None, pixel_to_meter=None
):
    """
    返回: dict{mask, pose, aff_mask, reason, ok}
    prox[i][j]=True 表示 i 侵入 j 的可供性；above[i][j]=True 表示 i 在 j 上方
    """
    me = masks[idx]
    N = len(masks)

    # A. 混合遮挡：有 k 满足 prox[k][idx] 且 above[idx][k]
    invaders_below = [k for k in range(N) if prox[k][idx] and above[idx][k]]
    if invaders_below:
        lower_u = union_masks_by_idx(masks, invaders_below)
        cut = cut_shadow_for_top(me, lower_u, expand_px, safety_px)
        if int(cut.sum()) >= min_area_px:
            pose = pose_from_mask(cut, pixel_to_meter=None)
            aff  = compute_affordance_mask(
                mask=cut, major_axis=pose['major'], minor_axis=pose['minor'],
                mean=np.array(pose['center'], np.float32),
                center_ratio=0.4, expand_pixels=expand_px
            )
            return dict(mask=cut, pose=pose, aff_mask=aff, reason="mixed", ok=True)

    # B. 互相靠近遮挡：存在 S 使 prox[idx][s] & prox[s][idx]
    mutual_neigh = [j for j in range(N) if prox[idx][j] and prox[j][idx]]
    if mutual_neigh:
        neigh_u = union_masks_by_idx(masks, mutual_neigh)
        cut = cut_overlap_for_mutual(me, neigh_u, expand_px, safety_px)
        if int(cut.sum()) >= min_area_px:
            pose = pose_from_mask(cut, pixel_to_meter=None)
            aff  = compute_affordance_mask(
                mask=cut, major_axis=pose['major'], minor_axis=pose['minor'],
                mean=np.array(pose['center'], np.float32),
                center_ratio=0.4, expand_pixels=expand_px
            )
            return dict(mask=cut, pose=pose, aff_mask=aff, reason="mutual_proximity", ok=True)

    # C. 单向靠近（我被别人侵入）
    invaders = [k for k in range(N) if prox[k][idx]]
    if invaders and orig_pose is not None:
        cut = carve_victim_side(me, orig_pose['center'], orig_pose['minor'], clearance_px=expand_px+safety_px)
        if int(cut.sum()) >= min_area_px:
            pose = pose_from_mask(cut, pixel_to_meter=None)
            aff  = compute_affordance_mask(
                mask=cut, major_axis=pose['major'], minor_axis=pose['minor'],
                mean=np.array(pose['center'], np.float32),
                center_ratio=0.4, expand_pixels=expand_px
            )
            return dict(mask=cut, pose=pose, aff_mask=aff, reason="oneway_proximity", ok=True)

    # D. 无需处理或处理失败：保持原 mask/pose
    pose = orig_pose if orig_pose is not None else pose_from_mask(me, pixel_to_meter=None)
    aff  = compute_affordance_mask(
        mask=me, major_axis=pose['major'], minor_axis=pose['minor'],
        mean=np.array(pose['center'], np.float32),
        center_ratio=0.4, expand_pixels=expand_px
    )
    return dict(mask=me.copy(), pose=pose, aff_mask=aff, reason="unchanged", ok=True)

# ===================== 测试主函数 =====================
def main_test():
    """
    目的：构造三个合成物体（椭圆）与三种遮挡关系，跑通“安全区域 + 姿态 + affordance”全流程。
    你也可以把这里替换成真实的 masks、prox、above 矩阵。
    """
    H, W = 360, 480
    masks = [np.zeros((H,W), np.uint8) for _ in range(3)]

    # 造三个椭圆：0=corn(细长), 1=potato(圆胖), 2=banana(偏细)
    cv2.ellipse(masks[0], (180,180), (70,25), 20, 0, 360, 255, -1)  # corn
    cv2.ellipse(masks[1], (240,190), (40,40), 0, 0, 360, 255, -1)   # potato
    cv2.ellipse(masks[2], (300,200), (60,20), -10, 0, 360, 255, -1) # banana

    # 构造 prox / above 关系（示例）：
    # prox[i][j]：i 侵入 j 的可供性；above[i][j]：i 在 j 上方
    N = len(masks)
    prox  = [[False]*N for _ in range(N)]
    above = [[False]*N for _ in range(N)]

    # 示例1：0 和 1 互相靠近；1 和 2 互相靠近；0 对 1 是深度上方（混合：0 prox→1 且 0 above 1）
    prox[0][1] = True; prox[1][0] = True
    prox[1][2] = True; prox[2][1] = True
    prox[0][1] = True  # 让 0 对 1 既靠近又上层 → 混合遮挡案例
    above[0][1] = True # 0 在 1 上方

    # 你的 compute_affordance_mask（引入你已有的实现）
    def compute_affordance_mask(mask, major_axis, minor_axis, mean, center_ratio=0.4, expand_pixels=15):
        # 这里直接复用你项目里的实现；此处写一个非常简化版以便可运行
        h, w = mask.shape
        aff = np.zeros_like(mask, np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0: return aff
        pts = np.stack([xs, ys], 1).astype(np.float32)
        rel = pts - mean
        proj_major = rel @ major_axis
        proj_minor = rel @ minor_axis
        # 主轴中间段
        proj_min, proj_max = proj_major.min(), proj_major.max()
        half = (proj_max - proj_min) * center_ratio / 2.0
        center = 0.5*(proj_max + proj_min)
        mid_mask = (proj_major >= center - half) & (proj_major <= center + half)
        # 副轴固定扩张
        band_mask = np.abs(proj_minor) <= float(expand_pixels)
        keep = mid_mask & band_mask
        sel = pts[keep].astype(np.int32)
        aff[sel[:,1], sel[:,0]] = 255
        # 小膨胀连通
        aff = cv2.dilate(aff, np.ones((5,5), np.uint8), 1)
        return aff

    # 先计算每个物体的原始姿态（供“单向靠近”时削带用）
    orig_poses = [pose_from_mask(m) for m in masks]

    results = []
    for idx in range(N):
        out = make_safe_region_for_object(
            idx=idx,
            masks=masks,
            prox=prox,
            above=above,
            compute_affordance_mask=compute_affordance_mask,
            expand_px=15, safety_px=3, min_area_px=200,
            orig_pose=orig_poses[idx],
            pixel_to_meter=None
        )
        pose = out['pose']
        results.append(dict(
            idx=idx, reason=out['reason'],
            center=(round(pose['center'][0],1), round(pose['center'][1],1)),
            yaw=round(pose['yaw'],1),
            width_px=round(pose['width_px'],1),
            safe_area=int(out['mask'].sum())
        ))

    print("=== Pipeline Results ===")
    for r in results:
        print(f"[obj {r['idx']}] reason={r['reason']}, center={r['center']}, yaw={r['yaw']}°, width={r['width_px']}px, safe_area={r['safe_area']}")

    vis = np.zeros((H, W, 3), np.uint8)
    colors = [(0,255,0),(255,0,0),(0,0,255)]
    for idx, r in enumerate(results):
        mask = r['mask'] if 'mask' in r else masks[idx]
        color = colors[idx % len(colors)]
        vis[mask > 0] = color
    cv2.imshow("safe regions", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    main_test()
