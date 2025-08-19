import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import numpy as np
import torch
import genesis as gs
from desk4 import create_scene  # 确保你有这个函数
import numpy as np

# 主控 DOFs（每个手指一个主控 DOF）
MAIN_GRIPPER_DOFS = [7,9]

def init_gripper_controller(xarm7):
    """
    Initialize gripper DOF control parameters: KP, KV, Force range
    """
    kp = [100.0] * len(MAIN_GRIPPER_DOFS)
    kv = [5.0] * len(MAIN_GRIPPER_DOFS)
    force_min = [-10.0] * len(MAIN_GRIPPER_DOFS)
    force_max = [10.0] * len(MAIN_GRIPPER_DOFS)

    xarm7.set_dofs_kp(kp, dofs_idx_local=MAIN_GRIPPER_DOFS)
    xarm7.set_dofs_kv(kv, dofs_idx_local=MAIN_GRIPPER_DOFS)
    xarm7.set_dofs_force_range(force_min, force_max, dofs_idx_local=MAIN_GRIPPER_DOFS)

def get_gripper_limits_from_entity(xarm7):
    """
    获取夹爪主控 DOF 的物理运动范围（下限和上限）
    """
    lower, upper = xarm7.get_dofs_limit(MAIN_GRIPPER_DOFS)
    print(lower.cpu().numpy(),"",upper.cpu().numpy())
    return lower.cpu().numpy(), upper.cpu().numpy()


def slow_gripper_control(xarm7, scene,start, end, steps=100,stop_on_contact=True, contact_entity=None):
    """
    平滑控制夹爪开合
    参数:
        steps: 插值步数（越大越平滑）
    """
    start_array,end_array  = get_gripper_limits_from_entity(xarm7)

    start_array[:]= start
    end_array[:] = end

    for t in range(steps):
        alpha = t / steps
        pos = (1 - alpha) * start_array + alpha * end_array
        xarm7.set_dofs_position(pos, dofs_idx_local=MAIN_GRIPPER_DOFS)
        scene.step()

        # # 🔍 可选：检测是否已夹到目标
        # if stop_on_contact and contact_entity is not None:
        #     contacts = xarm7.get_contacts(with_entity=contact_entity)
        #     if len(contacts) > 0:
        #         print(f"⛔️ Stopped early due to contact with '{contact_entity.name}' at step {t}")
        #         break

    final = xarm7.get_dofs_position(dofs_idx_local=MAIN_GRIPPER_DOFS)
    print(f"✅ Gripper move complete. Final position:", final.cpu().numpy())


def open_gripper(xarm7, scene,start = 0.0, end = 0.5, steps=100, stop_on_contact=True, contact_entity=None):
    slow_gripper_control(xarm7, scene, start,end, steps=steps, stop_on_contact=True, contact_entity=None)


def close_gripper(xarm7, scene, start = 0.5, end = 0.0, steps=100, stop_on_contact=True, contact_entity=None):
    slow_gripper_control(xarm7, scene, start,end, steps=steps, stop_on_contact=True, contact_entity=None)


    
def move_gripper_to(xarm7, scene, target_pos, steps=100):
    """
    平滑地将夹爪移动到指定的目标角度（绝对位置）
    
    参数:
        target_pos (float 或 list of float): 目标位置，长度应与 MAIN_GRIPPER_DOFS 相同
        steps: 插值步数
    """
    if isinstance(target_pos, (float, int)):
        target_pos = [target_pos] * len(MAIN_GRIPPER_DOFS)
    target_pos = np.array(target_pos, dtype=np.float32)

    current_pos = xarm7.get_dofs_position(dofs_idx_local=MAIN_GRIPPER_DOFS).cpu().numpy()

    for t in range(steps):
        alpha = t / steps
        pos = (1 - alpha) * current_pos + alpha * target_pos
        xarm7.set_dofs_position(pos, dofs_idx_local=MAIN_GRIPPER_DOFS)
        scene.step()

    print(f"🎯 Gripper moved to {target_pos}")




# === Main test function ===
if __name__ == "__main__":
    from gripper_utils import init_gripper_controller, open_gripper, close_gripper


    # 初始化 Genesis 场景
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui=True)

    # 等待渲染器稳定
    for _ in range(100):
        scene.step()



    init_gripper_controller(xarm7)

    # 在抓取前打开夹爪
    open_gripper(xarm7, scene)

    # 接近物体后慢慢闭合
    close_gripper(xarm7, scene)

    move_gripper_to(xarm7, scene, 0.5)
    move_gripper_to(xarm7, scene, 0)

    # import time
    # for _ in range(3):
    #     print("🚀 Closing...")
    #     close_gripper(xarm7, scene)
    #     time.sleep(0.5)
    #     print("🔁 Opening...")
    #     open_gripper(xarm7, scene)
    #     time.sleep(0.5)
