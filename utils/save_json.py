import json
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

    必须字段：
        - class: 类别名
        - coord_3d: 三维坐标

    可选字段：
        - gripper: 抓取宽度（米）
        - yaw: 抓取角度（度）
        - quat: 姿态四元数
        - occluded: 是否被遮挡
        - occlusion_type: 遮挡类型 ('depth' / 'proximity')
        - occluder: 遮挡者 ID
        - blocked_objects: 当前物体挡住的物体 ID
    """
    assert len(class_ids) == len(coords_3d), "class_ids 和 coords_3d 长度不一致"

    # 获取类别名映射
    if yolo_path:
        names = YOLO(yolo_path).names
    else:
        names = {}

    # 如果传入了遮挡信息，则初始化全局遮挡图
    occlusion_graph = {str(i): [] for i in range(len(class_ids))} if occlusion_info else None

    data = []
    for i, (cls_id, coord) in enumerate(zip(class_ids, coords_3d)):
        if coord is None:
            continue

        label = names.get(int(cls_id), f"id:{int(cls_id)}")
        entry = {
            "id": i,
            "class": label,
            "coord_3d": [round(coord[0], 3), round(coord[1], 3), round(coord[2], 3)]
        }

        # === 原始可选字段 ===
        if widths and i < len(widths):
            entry["gripper"] = round(widths[i], 2)
        if angles and i < len(angles):
            entry["yaw"] = round(angles[i], 2)
        if quaternions and i < len(quaternions):
            entry["quat"] = [round(q, 4) for q in quaternions[i]]

        # === 新增遮挡相关字段 ===
        if occlusion_info and i in occlusion_info:
            occ = occlusion_info[i]
            entry["occluded"] = not occ["ok"]
            entry["occlusion_type"] = occ["occ_type"]
            entry["occluder"] = occ["occluder"]
            entry["blocked_objects"] = []  # 当前物体挡住的物体ID

            # 如果该物体挡住了其他物体，更新全局遮挡图和自身blocked_objects
            if occ["occluder"] is not None:
                occlusion_graph[str(occ["occluder"])].append(str(i))

        data.append(entry)

    # 如果有遮挡信息，则保存全局遮挡图
    if occlusion_info:
        final_json = {
            "objects": data,
            "occlusion_graph": occlusion_graph
        }
    else:
        final_json = data  # 与原始行为保持一致

    # 写入JSON文件
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"✅ Scene info saved to: {output_path}")

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

    print(f"\n✅ Save successful: {output_path}")

if __name__ == "__main__":
    # 示例调用
    save_scene_info_to_json(
        yolo_path="checkpoints/best_yolo.pt",
        class_ids=[0, 1],
        coords_3d=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        output_path="output.json"
    )
