import os
import cv2
import json
import csv
import numpy as np
import pyrealsense2 as rs
from collections import defaultdict, deque
from ultralytics import YOLO
import math
from utils.yolo_utils import detect_fruits, merge_detections_by_depth_and_proximity
from utils.sam2_utils import segment_with_sam2
from utils.rotate import infer_grasp_angle_and_width, extract_grasp_infos
from utils.affordance_area import compute_affordance_mask, compute_affordance_striped_fullwidth, draw_affordance_on_image, visualize_graspables_fullmask, robust_center_depth
from utils.occluded import check_occlusion

def compute_occlusions(masks, depth_image, mclass_ids, class_names):
    edges_with_type = []
    for idx, mask in enumerate(masks):
        ok, j, occ_type, contact_pair = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=mask,       # 用物体mask本身
            center=np.mean(np.argwhere(mask), axis=0)[::-1],
            all_masks=masks,
            all_affordance_masks=masks,
            pixel_gap=5,
            debug=True,
            class_ids=mclass_ids,
            class_names=class_names,
            self_idx=idx
        )
        if ok and j is not None:
            edges_with_type.append((j, idx, occ_type))  # j遮挡了idx
    return edges_with_type

# ========== STEP 4. 判断互相遮挡并重新计算可抓区域 ==========
def recompute_affordance_for_mutual_occlusions(
    masks, depth_image, mclass_ids, class_names, edges_with_type):
    # 构建有向图
    forward = defaultdict(set)
    for u, v, t in edges_with_type:
        forward[u].add(v)

    # 找出互相遮挡的 pairs
    mutual_pairs = []
    for u, vs in forward.items():
        for v in vs:
            if u in forward and v in forward[u] and u in forward.get(v, set()):
                mutual_pairs.append((u, v))

    # 初始化可抓区域结果
    affordance_masks = [np.zeros_like(m, dtype=np.uint8) for m in masks]
    grasp_angles, widths, major_axes, minor_axes = [], [], [], []

    for i, m in enumerate(masks):
        # 如果参与互相遮挡，重新计算 affordance 区域
        if any(i in pair for pair in mutual_pairs):
            afford_mask, _ = compute_affordance_striped_fullwidth(
                idx=i,
                masks=masks,
                depth_image=depth_image,
                class_ids=mclass_ids,
                class_names=class_names,
                major_axes=major_axes,
                minor_axes=minor_axes,
                widths=widths
            )
            affordance_masks[i] = afford_mask
        else:
            affordance_masks[i] = masks[i]  # 不参与互相遮挡，直接用整mask

        # 重新计算抓取信息
        angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(
            affordance_masks[i], draw_on=None, object_index=i)
        grasp_angles.append(angle)
        widths.append(width)
        major_axes.append(major_axis)
        minor_axes.append(minor_axis)

    return affordance_masks, grasp_angles, widths, major_axes, minor_axes

def draw_mask_boundaries(image, masks, color=(0, 255, 0), thickness=2, ids=None, line_pairs=None):
    vis_image = image.copy()
    for idx, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(vis_image, contours, -1, color, thickness)
        if ids is not None:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(vis_image, str(ids[idx]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if line_pairs:
        for pt1, pt2 in line_pairs:
            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)
    return vis_image

def build_occlusion_graph(num_nodes, edges_with_type):
    adj = defaultdict(set)
    etypes = {}
    indeg = {i: 0 for i in range(num_nodes)}
    for u, v, t in edges_with_type:
        if v not in adj[u]:
            adj[u].add(v)
            etypes[(u, v)] = t
            indeg[v] += 1
    return adj, etypes, indeg

def topo_layers(adj, indeg):
    indeg = indeg.copy()
    q = deque([i for i, d in indeg.items() if d == 0])
    layers, visited = [], set()
    while q:
        layer = list(q)
        layers.append(layer)
        for _ in range(len(layer)):
            u = q.popleft()
            visited.add(u)
            for v in adj.get(u, []):
                indeg[v] -= 1
        for v, d in list(indeg.items()):
            if d == 0 and v not in visited and v not in q:
                q.append(v)
    return layers

def plan_to_reach_target(adj, indeg, target):
    rev = defaultdict(set)
    for u, vs in adj.items():
        for v in vs:
            rev[v].add(u)

    need = set()
    stack = [target]
    while stack:
        v = stack.pop()
        for u in rev.get(v, []):
            if u not in need:
                need.add(u)
                stack.append(u)

    sub_indeg = {n: 0 for n in need | {target}}
    sub_adj = defaultdict(set)
    for u in need | {target}:
        for v in adj.get(u, []):
            if v in sub_indeg:
                if v not in sub_adj[u]:
                    sub_adj[u].add(v)
                    sub_indeg[v] += 1

    order = []
    dq = deque([n for n, d in sub_indeg.items() if d == 0 and n != target])
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in sub_adj.get(u, []):
            sub_indeg[v] -= 1
            if sub_indeg[v] == 0 and v != target:
                dq.append(v)
    return order

# ---------- 汇总成“banana 被 potato, tomato 遮挡” ----------
def summarize_occlusions(edges_with_type, class_ids, class_names):
    by_blocked = {}
    for u, v, t in edges_with_type:
        by_blocked.setdefault(v, []).append((u, t))

    if not by_blocked:
        return ["无遮挡关系"], ["无遮挡关系"]

    def zh_type(t):
        return "上层" if t == "depth" else "靠近"

    lines_simple, lines_with_type = [], []
    for v, lst in sorted(by_blocked.items(), key=lambda kv: kv[0]):
        v_name = class_names[class_ids[v]]
        occluders_names = [class_names[class_ids[u]] for (u, _) in lst]
        lines_simple.append(f"{v_name} 被 {', '.join(occluders_names)} 遮挡")

        parts = [f"[{class_names[class_ids[u]]}:{zh_type(t)}]" for (u, t) in lst]
        lines_with_type.append(f"{v_name} 被 {', '.join(parts)} 遮挡")
    return lines_simple, lines_with_type

# ---------- 主程序 ----------
if __name__ == "__main__":
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    # path_base = "imgs/img_from_camera"
    path_base = "test_imgs/img0"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"

    # 也可从相机采集（已保留模板）
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # pipeline.start(config)
    # align = rs.align(rs.stream.color)
    # for _ in range(10): pipeline.wait_for_frames()
    # frames = align.process(pipeline.wait_for_frames())
    # color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
    # color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())
    # cv2.imwrite(img_path, color_image)
    # np.savez_compressed(npz_path, color=color_image, depth=depth_image)
    # pipeline.stop()

    if not os.path.exists(img_path) or not os.path.exists(npz_path):
        raise FileNotFoundError("缺少图像或深度文件，请检查 img_path/npz_path")

    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names  # dict: id->name

    color_image = cv2.imread(img_path)
    data = np.load(npz_path)
    depth_image = data["depth"]

    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)

    mboxes, mclass_ids, mcenters, groups = merge_detections_by_depth_and_proximity(
        boxes=boxes,
        class_ids=class_ids,
        centers_xy=yolo_centers,
        depth_image=depth_image.astype(np.uint16),
        pixel_dist_thresh=200,
        depth_diff_thresh=20,
        same_class_only=True,
        win=3
    )

    masks, sam_centers, result_path = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=mboxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )

    # 计算 affordance 区域
    image_with_aff = image_rgb.copy()
    affordance_masks = []
    for i, m in enumerate(masks):
        grasp_angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)
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
        image_with_aff = draw_affordance_on_image(image_with_aff, aff_mask, color=(0, 255, 0), alpha=0.4)
    cv2.imwrite("output_affordance_overlay.png", cv2.cvtColor(image_with_aff, cv2.COLOR_RGB2BGR))
    
    grasp_angles, widths, major_axes, minor_axes = [], [], [], []
    for i, m in enumerate(masks):
        angle, width, _, major_axis, minor_axis = infer_grasp_angle_and_width(m, draw_on=image_rgb, object_index=i)
        grasp_angles.append(angle)
        widths.append(width)
        major_axes.append(major_axis)
        minor_axes.append(minor_axis)

    # grasp_infos, grasp_img_path = extract_grasp_infos(
    #     image_rgb=image_rgb,
    #     masks=masks,
    #     pixel_to_meter=0.0005,
    #     output_path=f"{path_base}_grasp_visual.png"
    # )
    # 遮挡检测与边收集
    occlusion_line_pairs = []
    edges_with_type = []  # (occluder, blocked, type)
    for idx, mask in enumerate(masks):
        ok, j, occ_type, contact_pair = check_occlusion(
            depth_image=depth_image,
            mask=mask,
            affordance_mask=affordance_masks[idx],
            center=sam_centers[idx],
            all_masks=masks,
            all_affordance_masks=affordance_masks,
            debug=True,
            class_ids=mclass_ids,     # 注意与 mboxes 对齐后的 ids
            class_names=class_names,
            self_idx=idx
        )
        if ok and j is not None:
            edges_with_type.append((j, idx, occ_type))
            p1 = tuple(map(int, sam_centers[j]))
            p2 = tuple(map(int, sam_centers[idx]))
            occlusion_line_pairs.append((p1, p2))

    # 可视化边界与连线
    vis_img = draw_mask_boundaries(
        image_rgb,
        masks,
        ids=list(range(len(masks))),
        line_pairs=occlusion_line_pairs
    )
    os.makedirs("imgs", exist_ok=True)
    cv2.imwrite("imgs/occluded.png", vis_img)

        # === 用整物体mask计算可抓取区域（减去所有其他物体影响，保留最大连通域）===
    graspables, ginfos = [], []
    for i in range(len(masks)):
        g, info = compute_affordance_striped_fullwidth(
            idx=i,
            masks=masks,
            depth_image=depth_image.astype(np.uint16),
            class_ids=mclass_ids,
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
        graspables.append(g)
        ginfos.append(info)

    os.makedirs("imgs", exist_ok=True)
    visualize_graspables_fullmask(
        image_rgb=image_rgb,
        masks=masks,
        graspables=graspables,
        class_ids=mclass_ids,
        class_names=class_names,
        out_path="test_imgs/graspables_fullmask.png"
    )


    # # 构建图、分层、计划
    # N = len(masks)
    # adj, etypes, indeg = build_occlusion_graph(N, edges_with_type)

    # print("===== Occlusion Edges (u -> v) =====")
    # for u, vs in adj.items():
    #     for v in vs:
    #         name_u = class_names[mclass_ids[u]]
    #         name_v = class_names[mclass_ids[v]]
    #         t = etypes[(u, v)]
    #         zh = "上层" if t == 'depth' else "靠近"
    #         print(f"{u}({name_u}) -> {v}({name_v}) [{zh}]")

    # layers = topo_layers(adj, indeg)
    # print("===== Layers =====")
    # for k, layer in enumerate(layers, 1):
    #     labels = [f"{i}({class_names[mclass_ids[i]]})" for i in layer]
    #     print(f"Layer {k}: {labels}")

    # # 汇总“banana 被 potato, tomato 遮挡”
    # simple_lines, typed_lines = summarize_occlusions(edges_with_type, mclass_ids, class_names)
    # print("===== 遮挡汇总（简洁）=====")
    # for s in simple_lines: print(s)
    # print("===== 遮挡汇总（含类型）=====")
    # for s in typed_lines: print(s)

    # # 可选：导出 JSON 和 CSV
    # result_json = {
    #     "nodes": [
    #         {"idx": i, "class": class_names[mclass_ids[i]], "center": list(map(float, sam_centers[i]))}
    #         for i in range(N)
    #     ],
    #     "edges": [{"u": u, "v": v, "type": t} for (u, v, t) in edges_with_type],
    #     "layers": layers,
    #     "summary_simple": simple_lines,
    #     "summary_typed": typed_lines
    # }
    # with open("occlusion_graph.json", "w", encoding="utf-8") as f:
    #     json.dump(result_json, f, ensure_ascii=False, indent=2)


    # # 示例：以最后一个为目标，生成移除计划
    # if N > 0:
    #     target_idx = N - 1
    #     plan = plan_to_reach_target(adj, indeg, target_idx)
    #     plan_labels = [f"{i}({class_names[mclass_ids[i]]})" for i in plan] + [f"{target_idx}({class_names[mclass_ids[target_idx]]})"]
    #     print("===== Step Plan to reach target =====")
    #     print(" -> ".join(plan_labels))
