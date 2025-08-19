import json
import os

import torch
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from utils.desk4 import create_scene 
from xarm7.utils.gripper_utils import init_gripper_controller, close_gripper, open_gripper,move_gripper_to


if __name__ == "__main__":
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui=True)

    # lemon = fruits["lemon"]
    all_dof = np.arange(13)
    motors_dof = np.arange(7)
    gripper_dof = np.arange(7, 13)
    fingers_dof = np.arange(11,13)
    end_effector = xarm7.get_link("xarm_gripper_base_link")    

    # 2. PID & 力矩限制设置
    xarm7.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=motors_dof )
    xarm7.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=motors_dof )
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([ 87,  87,  87,  87,  12,  12,  12]),
        dofs_idx_local=motors_dof )

    pos_init = np.array([0, 0, 0, 1, 0, 0.5, 0])

    print("等待物理系统稳定...")

    for _ in range(100):
        scene.step()

    for i in range(100):
        xarm7.set_dofs_position(pos_init, motors_dof)
        scene.step()   
    print("初始姿态完成")

    # obj_pos = np.array([0.2, 0.03, 0.38])
    obj_pos = np.array([0.16, 0.05, 0.38])
    offset = np.array([0.053, -0.000, 0.15])
    quat = np.array([0, 0.5, 1, 0])
    # quat = quat / np.linalg.norm(quat)

    pos1=obj_pos + offset + np.array([0.0, 0.0, 0.15])

    # init_gripper_controller(xarm7) 
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        # pos=obj_pos + offset + np.array([0.0, 0.0, 0.15]),
        pos = pos1,
        quat = quat
    )

    qpos[7:] = 0.02
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)

    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step() # 报错

    for i in range(100):
        scene.step()
    print("预抓取位置")

    # reach
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.03]),
        # pos = obj_pos + offset,
        quat=quat
    )
    # print(lemon.get_pos())
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], motors_dof)
        scene.step()
    for i in range(100):
        scene.step()
    print("reach")

    # grasp
    qpos[7:] = 0.4
    xarm7.control_dofs_position(qpos)
    xarm7.control_dofs_force(np.array([-1, -1]), fingers_dof)
    scene.step()
    for i in range(100):
        scene.step()
    print("grasp")

    # lift
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos1,
        quat=quat
    )
    xarm7.control_dofs_position(qpos[:7], motors_dof)
    for i in range(100):
        scene.step()
    print("lift")

    # move_gripper_to(xarm7, scene, 0.5)
    # move_gripper_to(xarm7, scene, 0)
