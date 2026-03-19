import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import json
import numpy as np
import genesis as gs
from utils.desk4 import create_scene
from utils.gripper_utils import init_gripper_controller, open_gripper, close_gripper

def report_contacts(robot):
    contacts = robot.get_contacts()
    if not contacts:
        print("⚠️ 没有检测到接触")
    else:
        print(f"✅ 共检测到 {len(contacts)} 个接触点")
        for i, c in enumerate(contacts):
            pos = c.position
            impulse = c.impulse
            print(f"  Contact {i+1}: 位置 = {pos}, 冲量 = {impulse}")

def load_coord_from_json(json_path, target_class):
    with open(json_path, "r") as f:
        data = json.load(f)
    for obj in data:
        if obj["class"] == target_class:
            return np.array(obj["coord_3d"])
    raise ValueError(f"❌ JSON 中找不到目标类 '{target_class}'")

def pick_by_coord(obj_pos, scene, xarm7, ee, motors_dof, pre_grasp_height=0.10, lift_height=0.25):
    quat = np.array([0, 0, 1, 0])

    # 上方预抓取位置
    above_pos = obj_pos + np.array([0, 0, pre_grasp_height])
    qpos = xarm7.inverse_kinematics(link=ee, pos=above_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[motors_dof], motors_dof)
        scene.step()
    print("✅ 已移动到物体上方")

    # 打开夹爪
    open_gripper(xarm7, scene, steps=50)

    # 向下贴近
    grasp_pos = obj_pos + np.array([0, 0, 0.01])
    qpos = xarm7.inverse_kinematics(link=ee, pos=grasp_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[motors_dof], motors_dof)
        scene.step()
    print("✅ 已贴近物体")

    # 抓取前检测接触
    report_contacts(xarm7)

    # 闭合夹爪
    close_gripper(xarm7, scene, steps=60)

    # 抬起
    lift_pos = obj_pos + np.array([0, 0, lift_height])
    qpos = xarm7.inverse_kinematics(link=ee, pos=lift_pos, quat=quat)
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[motors_dof], motors_dof)
        scene.step()
    print("✅ 已抬起物体")

    # 再次检测接触
    report_contacts(xarm7)

def pick_from_json(json_path, target_class):
    print(f"🎯 抓取目标: {target_class}")
    scene, xarm7, _, _, camera = create_scene(enable_gui=True)
    init_gripper_controller(xarm7)
    motors_dof = np.arange(7)
    ee = xarm7.get_link("xarm_gripper_base_link")

    # 获取 3D 位置
    obj_pos = load_coord_from_json(json_path, target_class)
    pick_by_coord(obj_pos, scene, xarm7, ee, motors_dof)

if __name__ == "__main__":
    pick_from_json("imgs/sim_fruit_from_camera.json", target_class="potato")
