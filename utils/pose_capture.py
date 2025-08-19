import numpy as np
import cv2
import os
from utils.desk4 import create_scene

def capture(img_path, npz_path):
    """
    控制 xArm7 到达指定姿态，并保存 RGB 图像和点云信息

    参数:
        scene: Genesis 场景对象
        xarm7: xArm7 机械臂对象
        camera: 相机对象
        enable_gui (bool): 是否开启渲染
        img_path (str): 保存 RGB 图像的路径
        npz_path (str): 保存点云的路径

    返回:
        dict: 包含 'img_path' 和 'npz_path'
    """
    enable_gui=True
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui)
    print("等待物理系统稳定...")
    for _ in range(100):
        scene.step()

    jnt_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx = [xarm7.get_joint(name).dof_idx_local for name in jnt_names]

    # PID 和力矩限制
    xarm7.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]), dofs_idx_local=dofs_idx)
    xarm7.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]), dofs_idx_local=dofs_idx)
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12]),
        dofs_idx_local=dofs_idx
    )

    # 初始姿态
    pos_init = np.array([0, 0, 0, 1, 0, 0.5, 0])
    for _ in range(100):
        scene.step()
    for _ in range(100):
        xarm7.set_dofs_position(pos_init, dofs_idx)
        scene.step()

    # 获取末端执行器和夹爪位置
    end_effector = xarm7.get_link("xarm_gripper_base_link")
    base_pos = end_effector.get_pos()
    left_finger_pos = xarm7.get_link("left_finger").get_pos()
    right_finger_pos = xarm7.get_link("right_finger").get_pos()
    gripper_offset = (left_finger_pos + right_finger_pos) / 2 - base_pos

    # 设置目标位置和姿态（Z轴向下）
    target_pos = np.array([0.20, 0.0, 0.85])
    target_quat = np.array([0, 0, 1, 0])

    # 求解逆解 + 设置夹爪张开
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)
    qpos[7:] = 0.04

    # 路径规划 + 执行
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=200)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step()
        if enable_gui:
            camera.render(rgb=True)

    for _ in range(50):
        scene.step()
    print("✅ 控制结束")

    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if enable_gui:
        # 保存 RGB 图像
        rgb_img, _, _, _ = camera.render(rgb=True)
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, rgb_bgr)
        print(f"✅ 图像已保存到: {img_path}")

        # 保存点云
        pointcloud, mask_idx = camera.render_pointcloud(world_frame=True)[:2]
        mask = np.zeros(pointcloud.shape[:2], dtype=bool)
        mask[mask_idx] = True
        np.savez(npz_path, pointcloud=pointcloud, mask=mask)
        print(f"✅ 点云和掩码已保存到: {npz_path}")

    return {"img_path": img_path, "npz_path": npz_path}
