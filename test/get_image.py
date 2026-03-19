import pyrealsense2 as rs
import numpy as np
import cv2
import os

def visualize_depth_gamma(depth_image, min_d=350, max_d=550, gamma=0.6):
    depth = depth_image.astype(np.float32)
    depth = np.clip(depth, min_d, max_d)
    depth_norm = (depth - min_d) / (max_d - min_d + 1e-5)
    depth_gamma = np.power(depth_norm, gamma)
    depth_scaled = (depth_gamma * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(255 - depth_scaled, cv2.COLORMAP_JET)
    return colorized


# 设置保存文件夹
save_dir = "test_imgs"
os.makedirs(save_dir, exist_ok=True)

# 查找已有文件，确定下一个 i 值
existing = [f for f in os.listdir(save_dir) if f.startswith("img") and f.endswith(".png")]
indices = [int(f[3:].split('.')[0]) for f in existing if f[3:].split('.')[0].isdigit()]
i = max(indices) + 1 if indices else 0

# 构建保存文件名（不带扩展名）
save_base = os.path.join(save_dir, f"img{i}")

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

# 预热
for _ in range(10):
    pipeline.wait_for_frames()

# 获取对齐帧
frames = pipeline.wait_for_frames()
aligned = align.process(frames)
color_frame = aligned.get_color_frame()
depth_frame = aligned.get_depth_frame()

if not color_frame or not depth_frame:
    print("[ERROR] 获取帧失败！")
    exit(1)

# 转为数组
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# 应用
depth_colormap = visualize_depth_gamma(depth_image, min_d=400, max_d=550, gamma=0.6)

# 保存图像和数据
cv2.imwrite(f"{save_base}.png", color_image)
cv2.imwrite(f"{save_base}_depth.png", depth_colormap)
np.savez_compressed(f"{save_base}.npz", color=color_image, depth=depth_image)

print(f"[INFO] 成功保存图像为 img{i}")
print(f"       彩色图像: {save_base}.png")
print(f"       深度图像: {save_base}_depth.png")
print(f"       数据文件: {save_base}.npz")

pipeline.stop()
