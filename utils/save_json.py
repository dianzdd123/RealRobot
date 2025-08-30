import json
from ultralytics import YOLO

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
    occlusion_info=None  # 新增：遮挡信息
):
    """
    保存场景中每个物体的信息为 JSON 文件。
    新增字段：
        occlusions: 该物体被哪些物体遮挡，列表形式，元素包含：
            - by_id: 遮挡者的索引
            - by_class: 遮挡者的类别名
            - type: 遮挡类型 ("depth" | "proximity")
    """
    assert len(class_ids) == len(coords_3d), "class_ids 和 coords_3d 长度不一致"

    # 加载 YOLO 类别映射
    names = YOLO(yolo_path).names if yolo_path else {}

    data = []
    for i, (cls_id, coord) in enumerate(zip(class_ids, coords_3d)):
        if coord is None:
            continue

        entry = {
            "id": i,
            "class": names.get(int(cls_id), f"id:{int(cls_id)}"),
            "coord_3d": [round(coord[0], 3), round(coord[1], 3), round(coord[2], 3)]
        }

        # 可选字段
        if widths and i < len(widths):
            entry["gripper"] = round(widths[i], 4)
        if angles and i < len(angles):
            entry["yaw"] = round(angles[i], 2)
        if quaternions and i < len(quaternions):
            entry["quat"] = [round(q, 4) for q in quaternions[i]]

        # 遮挡信息
        entry["occlusions"] = []
        if occlusion_info and i in occlusion_info:
            for occ in occlusion_info[i]:
                entry["occlusions"].append({
                    "by_id": occ["by_id"],
                    "by_class": names.get(int(class_ids[occ["by_id"]]), f"id:{int(class_ids[occ['by_id']])}"),
                    "type": occ["type"]
                })

        data.append(entry)

    # 写入 JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ 场景信息及遮挡关系已保存到: {output_path}")


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
