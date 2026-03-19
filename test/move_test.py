import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
import numpy as np
import genesis as gs
from desk4 import create_scene

########################## main ##########################
if __name__ == "__main__":
    scene, xarm7, fruits, bins, camera = create_scene()
    print("✅ 场景创建成功，开始控制 xArm7...")

    jnt_names = [
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6", "joint7"
    ]
    dofs_idx = [xarm7.get_joint(name).dof_idx_local for name in jnt_names]

    # 2. PID & 力矩限制设置
    xarm7.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=dofs_idx
    )
    xarm7.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=dofs_idx
    )
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([ 87,  87,  87,  87,  12,  12,  12]),
        dofs_idx_local=dofs_idx
    )

    # 3. 定义初始和目标关节角
    pos_init   = np.array([0, 0, 0, 1, 0, 0.5, 0])
    for _ in range(100):
        scene.step()

    # 6. 重置到初始姿态
    for i in range(100):
        xarm7.set_dofs_position(pos_init, dofs_idx)
        scene.step()

    # # camera.render(rgb=True)

    # 设置末端执行器为夹爪基座
    end_effector = xarm7.get_link("xarm_gripper_base_link")
    # 设置显示该 link 的坐标轴

    end_pos = end_effector.get_pos().cpu().numpy()
    print("末端执行器位置：",end_pos)

    # 设置目标位置和方向（竖直向下）
    target_pos = np.array([0.18, 0.04, 0.55])
    target_quat = np.array([0, 0, 1, 0])  # Z轴朝下，姿态可微调

    # 使用逆运动学求解目标姿态的关节位置
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
    )

    # 设置夹爪张开（假设末6个自由度是夹爪）
    qpos[7:] = 0.04

    # 生成运动路径
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=200)

    # 执行路径
    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step()
        camera.render(
            rgb = True,
            # depth = True,
        )
    # 等待稳定
    for _ in range(500):
        scene.step()
        camera.render(
            rgb = True,
            # depth = True,
        )
    # control_xarm7(scene, xarm7, cam_attachment, camera)
    # print(camera.transform)
    print("✅ 控制结束")
