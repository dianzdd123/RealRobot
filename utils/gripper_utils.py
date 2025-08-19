import numpy as np

class GripperController:
    def __init__(self, xarm7, scene, main_dofs=(7, 9), max_close=0.85):
        """
        初始化夹爪控制器。

        参数:
            xarm7: 机械臂对象
            scene: Genesis 场景对象（用于 step）
            main_dofs: 夹爪两个主控 DOF 索引
            max_close: 最大闭合角度（单位取决于模型，通常是0~0.85）
        """
        self.xarm7 = xarm7
        self.scene = scene
        self.main_dofs = main_dofs
        self.max_close = max_close

        # 初始化控制参数
        kp = [100.0] * len(self.main_dofs)
        kv = [5.0] * len(self.main_dofs)
        force_min = [-10.0] * len(self.main_dofs)
        force_max = [10.0] * len(self.main_dofs)

        xarm7.set_dofs_kp(kp, dofs_idx_local=self.main_dofs)
        xarm7.set_dofs_kv(kv, dofs_idx_local=self.main_dofs)
        xarm7.set_dofs_force_range(force_min, force_max, dofs_idx_local=self.main_dofs)

    def open(self, steps=100):
        """完全打开夹爪"""
        self._interpolate_move(0.0, self.max_close, steps)

    def close(self, percent=1.0, steps=100):
        """
        按比例闭合夹爪。
        参数:
            percent: 0.0（全开）到 1.0（完全闭合）
        """
        percent = max(0.0, min(1.0, percent))
        target_angle = self.max_close * percent
        self._interpolate_move(self.max_close, target_angle, steps)

    def move_to(self, target_angle, steps=100):
        """
        将夹爪移动到指定目标位置（绝对角度）
        """
        self._interpolate_move(None, target_angle, steps)

    def _interpolate_move(self, start, end, steps=100):
        """
        内部函数：平滑移动夹爪从 start 到 end。
        如果 start=None，则从当前位置开始。
        """
        if isinstance(end, (float, int)):
            end = [end] * len(self.main_dofs)

        if start is None:
            start = self.xarm7.get_dofs_position(dofs_idx_local=self.main_dofs).cpu().numpy()
        elif isinstance(start, (float, int)):
            start = [start] * len(self.main_dofs)

        start = np.array(start)
        end = np.array(end)

        for t in range(steps):
            alpha = t / steps
            pos = (1 - alpha) * start + alpha * end
            self.xarm7.set_dofs_position(pos, dofs_idx_local=self.main_dofs)
            self.scene.step()

        print(f"✅ Gripper moved to {end}")
