import os, json
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import numpy as np
import genesis as gs
from utils.desk4 import create_scene
from utils.gripper_utils import init_gripper_controller, open_gripper, close_gripper

# name of the JSON file you generated from sim_fruit_from_camera
JSON_PATH = "imgs/sim_fruit_from_camera.json"

def match_entities_to_json(fruits, objects):
    """Assumes create_scene() loads fruits in the same order as your JSON list."""
    if len(fruits) != len(objects):
        raise RuntimeError(f"Got {len(fruits)} entities but {len(objects)} JSON entries")
    return dict(zip([o["class"] for o in objects], fruits))

def report_contacts(robot, entity):
    """Prints contact info between robot and a single entity."""
    info = robot.get_contacts(with_entity=entity)
    valid = info["valid_mask"] if "valid_mask" in info else None
    print("  contact pairs:", len(info["geom_a"]) if "geom_a" in info else "n/a",
          "valid_mask:", valid)
def report_contacts(robot, target_obj):
    """
    报告机器人和目标物体之间的接触信息
    """
    contacts = robot.get_contacts(with_entity=target_obj)
    if not contacts:
        print("⚠️ 没有检测到接触")
    else:
        print(f"✅ 共检测到 {len(contacts)} 个接触点")
        for i, c in enumerate(contacts):
            pos = c.position
            impulse = c.impulse
            print(f"  Contact {i+1}: 位置 = {pos}, 冲量 = {impulse}")


def debug_grasp(entity_name="potato", pre_grasp_height=0.10, slow_steps=120):
    """
    用于测试夹爪是否能够正确抓住物体
    """
    # ✅ 初始化场景
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui=True)
    init_gripper_controller(xarm7)

    # ✅ 获取目标物体实体
    target_obj = None
    for fruit in fruits:
        if entity_name in fruit.name:
            target_obj = fruit
            break
    if target_obj is None:
        raise RuntimeError(f"实体名 '{entity_name}' 没找到，当前场景中有： {[f.name for f in fruits]}")

    # ✅ 获取物体位置
    obj_pos = target_obj.get_pose()[0]
    above_pos = obj_pos + np.array([0.0, 0.0, pre_grasp_height])
    quat = np.array([0, 0, 1, 0])

    # ✅ 获取末端执行器链接
    ee = xarm7.get_link("xarm_gripper_base_link")
    dofs_idx = np.arange(7)

    # ✅ 移动到物体上方
    qpos = xarm7.inverse_kinematics(link=ee, pos=above_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=80)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], dofs_idx)
        scene.step()
    print("→ Moved to pre-grasp above", entity_name)

    # ✅ 打开夹爪
    open_gripper(xarm7, scene, steps=50)
    print("→ Opened gripper")

    # ✅ 向下贴近物体
    close_pos = obj_pos + np.array([0.0, 0.0, 0.02])
    qpos = xarm7.inverse_kinematics(link=ee, pos=close_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=slow_steps)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], dofs_idx)
        scene.step()
    print("→ Moved down to grasp")

    # ✅ 抓取前检测接触
    print("Before closing:")
    report_contacts(xarm7, target_obj)

    # ✅ 闭合夹爪
    close_gripper(xarm7, scene, steps=60)
    print("→ Closed gripper")

    # ✅ 抬起
    up_pos = obj_pos + np.array([0.0, 0.0, 0.25])
    qpos = xarm7.inverse_kinematics(link=ee, pos=up_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=80)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], dofs_idx)
        scene.step()
    print("→ Lifted object")

    # ✅ 再次检测接触
    print("After closing and lifting:")
    report_contacts(xarm7, target_obj)

if __name__ == "__main__":
    # change “potato” to “banana” or any other class in your JSON
    debug_grasp(entity_name="potato", pre_grasp_height=0.10, slow_steps=120)
