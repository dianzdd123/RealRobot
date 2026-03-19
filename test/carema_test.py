import os
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'glx'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import genesis as gs

# 初始化Genesis
gs.init(backend=gs.gpu)

# 创建场景
scene = gs.Scene(show_FPS=False)

# 添加地面
plane = scene.add_entity(gs.morphs.Plane())

# 加载机械臂 (xArm7)
xarm7 = scene.add_entity(gs.morphs.MJCF(
    file="models/ufactory_xarm7/xarm7.xml",
    pos=(0.0, 0.0, 0.0)),
    # vis_mode="collision"
    vis_mode="visual"
)

camera = scene.add_camera(
    res=(1640, 1640),
    pos=(0, 0.0, 1.0),
    lookat=(0.0, 0.0, 0.0),
    fov=60,
    GUI=True,  # 开GUI窗口实时显示
)

# 选取需要绑定的末端Link
gripper_link = xarm7.get_link("xarm_gripper_base_link")

# 搭建场景
scene.build()

T = np.array([
    [ 0, -1,  0, -0.08],  # X': -Y
    [ 1,  0,  0,  0   ],  # Y':  X
    [ 0,  0, -1,  0.02],  # ✅ Z': -Z ✅ 向下看 ✅ 光照正常
    [ 0,  0,  0,  1   ]
])

camera.attach(rigid_link=gripper_link, offset_T = T)

# 初始化机械臂姿态
init_qpos = xarm7.get_qpos().cpu().numpy()
moving_joint_indices = [1, 3, 5]

# # 开始录视频
# camera.start_recording()

# 运行仿真
for i in range(5000):
    t = i / 60.0
    new_qpos = init_qpos.copy()
    for j, idx in enumerate(moving_joint_indices):
        new_qpos[idx] += 0.3 * np.sin(t + j * np.pi / 3)

    xarm7.set_qpos(new_qpos)
    scene.step()

    # # 刷新相机位置
    # attachment.update()

    # 渲染并保存
    camera.render()

# # 结束录制
# camera.stop_recording(save_to_filename='robot_camera_follow.mp4', fps=60)

print("✅ 录制完成，视频保存为 'robot_camera_follow.mp4'")
