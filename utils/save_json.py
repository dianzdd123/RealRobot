import os
import json
import numpy as np
from ultralytics import YOLO

def save_scene_info_to_json(
    class_ids,
    coords_3d,
    output_path,
    yolo_path=None,
    widths=None,
    angles=None,
    quaternions=None,
    occlusion_info=None
):
    """
    保存场景中每个物体的信息为 JSON 文件。
    兼容：
      - coords_3d 可含 None（将被跳过）
      - occlusion_info 支持 [(j, type)] 或 [{'by_id': j, 'type': t}] 或 [{'by': j, 'type': t}]
    """

    # 载入类别名称映射（若不给 yolo_path 就用空映射）
    names = {}
    if yolo_path:
        try:
            names = YOLO(yolo_path).names or {}
        except Exception as e:
            print(f"[WARN] 载入 YOLO 名称失败：{e}，将使用 id:id 映射")

    def id_to_name(cid):
        try:
            if isinstance(names, dict) and isinstance(cid, int) and cid in names:
                return str(names[cid])
            return f"id:{int(cid)}"
        except Exception:
            return str(cid)

    # 统一长度，避免索引越界
    N = len(class_ids)
    def safe_len(lst):
        return 0 if lst is None else len(lst)

    # 将可选列表补足或裁剪到 N（不对 coords_3d 操作，coords_3d 保持原样）
    def fix_len(arr, n, fill=None):
        if arr is None:
            return [fill] * n
        arr = list(arr)
        if len(arr) < n:
            arr = arr + [fill] * (n - len(arr))
        elif len(arr) > n:
            arr = arr[:n]
        return arr

    widths       = fix_len(widths, N, fill=None)
    angles       = fix_len(angles, N, fill=None)
    quaternions  = fix_len(quaternions, N, fill=None)

    # 解析遮挡关系条目 -> (by_id:int, type:str)；无法解析返回 None
    def parse_occ_item(item):
        # 1) 元组/列表，如 (j, "depth") 或 [j, "proximity", ...] 取前两个
        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                return int(item[0]), str(item[1])
            return None
        # 2) 字典
        if isinstance(item, dict):
            # 兼容 by_id / by / idx / id
            j = item.get("by_id", item.get("by", item.get("idx", item.get("id"))))
            t = item.get("type", item.get("kind", item.get("mode")))
            if j is None or t is None:
                return None
            try:
                j = int(j)
            except Exception:
                return None
            return j, str(t)
        return None

    data = []
    skipped_none = 0
    out_of_range_occ = 0

    # 逐个目标写入
    for i in range(N):
        # coords_3d 可能长度与 N 不一致，这里做安全读取
        if i >= len(coords_3d):
            print(f"[WARN] coords_3d 长度不足：i={i}, len(coords_3d)={len(coords_3d)}，跳过该目标")
            continue

        coord = coords_3d[i]
        if coord is None:
            skipped_none += 1
            continue

        # 坐标规整
        try:
            x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        except Exception:
            print(f"[WARN] 坐标格式异常，跳过 i={i}, coord={coord}")
            continue

        cls_id = class_ids[i]
        entry = {
            "id": i,
            "class": id_to_name(int(cls_id)) if isinstance(cls_id, (int, np.integer)) or str(cls_id).isdigit() else id_to_name(cls_id),
            "coord_3d": [round(x, 3), round(y, 3), round(z, 3)]
        }

        # 可选字段
        if widths[i] is not None:
            try:
                entry["gripper"] = round(float(widths[i]), 4)
            except Exception:
                pass
        if angles[i] is not None:
            try:
                entry["yaw"] = round(float(angles[i]), 2)
            except Exception:
                pass
        if quaternions[i] is not None:
            try:
                entry["quat"] = [round(float(q), 4) for q in quaternions[i]]
            except Exception:
                pass

        # 遮挡信息（仅记录有效 by_id 且在范围内的）
        entry["occlusions"] = []
        if isinstance(occlusion_info, dict) and i in occlusion_info:
            rels = occlusion_info[i] or []
            for it in rels:
                parsed = parse_occ_item(it)
                if not parsed:
                    print(f"[WARN] 无法解析遮挡条目：obj={i}, item={it}")
                    continue
                by_id, occ_type = parsed
                if by_id < 0 or by_id >= N:
                    out_of_range_occ += 1
                    continue
                entry["occlusions"].append({
                    "by_id": by_id,
                    "by_class": id_to_name(int(class_ids[by_id])) if isinstance(class_ids[by_id], (int, np.integer)) or str(class_ids[by_id]).isdigit() else id_to_name(class_ids[by_id]),
                    "type": occ_type
                })

        data.append(entry)

    # 输出目录
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 场景信息及遮挡关系已保存到: {output_path}")
    if skipped_none:
        print(f"[INFO] 已跳过 {skipped_none} 个 None 坐标目标（未写入 JSON）")
    if out_of_range_occ:
        print(f"[INFO] 有 {out_of_range_occ} 条遮挡关系的 by_id 越界被忽略")


def save_json_from_detection(
    class_ids,
    centers,
    coords_3d,
    output_path,
    yolo_path=None,
    class_names=None,
    gripper_opens=None,
    gripper_closes=None,
    quaternions=None,
    angles=None  # 新增角度
):
    """
    通用 JSON 保存函数，兼容 YOLO/SAM 自动分割 + Grasp 信息

    参数:
        class_ids (List[int])
        centers (List[Tuple[int, int]])：2D 中心点坐标
        coords_3d (List[Tuple[float, float, float]])
        output_path (str)
        yolo_path (str, optional)
        class_names (Dict[int, str], optional)
        widths (List[float], optional): 抓取宽度（米）
        quaternions (List[List[float]], optional): 四元数姿态
        angles (List[float], optional): 抓取角度（deg）
    """
    # 1. 加载类别名映射
    if class_names is not None:
        names = class_names
    elif yolo_path is not None:
        names = YOLO(yolo_path).names
    else:
        names = {}

    assert len(class_ids) == len(centers) == len(coords_3d)

    data = []
    for i, (cls_id, center, coord) in enumerate(zip(class_ids, centers, coords_3d)):
        if coord is None:
            continue

        label = names.get(int(cls_id), f"id:{int(cls_id)}")

        entry = {
            "class": label,
            # "center": [int(center[0]), int(center[1])],
            "coord_3d": [round(float(coord[0]), 3), round(float(coord[1]), 3), round(float(coord[2]), 3)]
        }

        if gripper_opens and gripper_closes:
            entry["gripper_open"] = int(gripper_opens[i])
            entry["gripper_close"] = int(gripper_closes[i])

        if quaternions:
            entry["quat"] = [round(q, 4) for q in quaternions[i]]

        if angles:
            entry["yaw"] = round(angles[i], 2)

        data.append(entry)


        # 打印输出
        print(f"{entry['class']} @ {entry['coord_3d']}", end="")
        if "grasp_width" in entry:
            print(f", width={entry['grasp_width']}", end="")
        if "grasp_angle_deg" in entry:
            print(f", angle={entry['grasp_angle_deg']}", end="")
        if "grasp_quaternion" in entry:
            print(f", quat={entry['grasp_quaternion']}", end="")
        print()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n Save successful: {output_path}")

if __name__ == "__main__":
    # 示例调用
    save_scene_info_to_json(
        yolo_path="checkpoints/best_yolo.pt",
        class_ids=[0, 1],
        coords_3d=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        output_path="output.json"
    )
