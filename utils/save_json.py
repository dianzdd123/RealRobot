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
    occlusion_info=None,
    graspable_flags=None
):
    """Serializes comprehensive scene information including 3D coordinates and occlusion relations to JSON."""
    names = {}
    if yolo_path:
        try:
            names = YOLO(yolo_path).names or {}
        except Exception as e:
            print(f"[WARN] Failed to load YOLO names from {yolo_path}: {e}. Falling back to ID mapping.")

    def id_to_name(cid):
        """Maps class ID to human-readable name."""
        try:
            if isinstance(names, dict) and isinstance(cid, int) and cid in names:
                return str(names[cid])
            return f"id:{int(cid)}"
        except Exception:
            return str(cid)

    N = len(class_ids)

    def fix_len(arr, n, fill=None):
        """Ensures the list matches the required length by padding or trimming."""
        if arr is None:
            return [fill] * n
        arr = list(arr)
        if len(arr) < n:
            arr += [fill] * (n - len(arr))
        elif len(arr) > n:
            arr = arr[:n]
        return arr

    # Align all attribute lists to the number of detected objects
    widths = fix_len(widths, N)
    angles = fix_len(angles, N)
    quaternions = fix_len(quaternions, N)
    graspable_flags = fix_len(graspable_flags, N, fill=True)

    def parse_occ_item(item):
        """Parses various occlusion formats into a standard (ID, type) tuple."""
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return int(item[0]), str(item[1])
        if isinstance(item, dict):
            j = item.get("by_id", item.get("by", item.get("idx", item.get("id"))))
            t = item.get("type", item.get("kind", item.get("mode")))
            if j is not None and t is not None:
                try:
                    return int(j), str(t)
                except:
                    return None
        return None

    data = []
    skipped_none = 0
    out_of_range_occ = 0

    for i in range(N):
        if i >= len(coords_3d):
            print(f"[WARN] Insufficient 3D coordinates: i={i}, skipping object.")
            continue

        coord = coords_3d[i]
        if coord is None:
            skipped_none += 1
            continue

        try:
            x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        except Exception:
            print(f"[WARN] Invalid coordinate format for object {i}, skipping.")
            continue

        cls_id = class_ids[i]
        entry = {
            "id": i,
            "class": id_to_name(int(cls_id)) if isinstance(cls_id, (int, np.integer)) or str(cls_id).isdigit() else id_to_name(cls_id),
            "coord_3d": [round(x, 3), round(y, 3), round(z, 3)],
            "graspable": bool(graspable_flags[i])
        }

        # Optional attributes
        if widths[i] is not None:
            try: entry["gripper"] = round(float(widths[i]), 4)
            except: pass
        if angles[i] is not None:
            try: entry["yaw"] = round(float(angles[i]), 2)
            except: pass
        if quaternions[i] is not None:
            try: entry["quat"] = [round(float(q), 4) for q in quaternions[i]]
            except: pass

        # Process occlusion relations
        entry["occlusions"] = []
        if isinstance(occlusion_info, dict) and i in occlusion_info:
            rels = occlusion_info[i] or []
            for it in rels:
                parsed = parse_occ_item(it)
                if not parsed:
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Scene information saved to: {output_path}")
    if skipped_none:
        print(f"[INFO] Skipped {skipped_none} objects due to missing 3D coordinates.")

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
    angles=None
):
    """Saves detection and basic grasp data to a JSON file for baseline pipelines."""
    if class_names is not None:
        names = class_names
    elif yolo_path is not None:
        names = YOLO(yolo_path).names
    else:
        names = {}

    data = []
    for i, (cls_id, center, coord) in enumerate(zip(class_ids, centers, coords_3d)):
        if coord is None:
            continue

        label = names.get(int(cls_id), f"id:{int(cls_id)}")
        entry = {
            "class": label,
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

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Detection JSON saved to: {output_path}")