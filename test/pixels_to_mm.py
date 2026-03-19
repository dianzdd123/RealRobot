import pyrealsense2 as rs
import numpy as np
import cv2

def compute_pixel_to_meter(depth_frame, depth_intrin, pt0, pt1):
    x0, y0 = pt0
    x1, y1 = pt1

    d0 = depth_frame.get_distance(x0, y0)
    d1 = depth_frame.get_distance(x1, y1)

    if d0 == 0 or d1 == 0:
        raise ValueError("深度为0，可能测量点在无效区域")

    p0 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x0, y0], d0)
    p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x1, y1], d1)

    dist = np.linalg.norm(np.array(p0) - np.array(p1))  # 真实距离（米）
    pixel_length = np.linalg.norm(np.array(pt1) - np.array(pt0))  # 像素间距
    print(f"深度值 d0 = {d0:.4f} m, d1 = {d1:.4f} m")

    print(f"[调试] 像素间距: {pixel_length:.2f} px")
    print(f"[调试] 真实距离: {dist*1000:.4f} mm")

    return dist / pixel_length


if __name__ == "__main__":
    # 初始化 RealSense 流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        for _ in range(30):  # 等待自动曝光稳定
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 选择两个点：你可以换成你想测的位置
        pt0 = (345, 200)
        pt1 = (600, 200)

        # 可视化标记
        cv2.circle(color_image, pt0, 5, (0, 255, 0), -1)
        cv2.circle(color_image, pt1, 5, (0, 255, 0), -1)
        cv2.line(color_image, pt0, pt1, (0, 255, 255), 2)

        # 执行计算
        pixel_to_meter = compute_pixel_to_meter(depth_frame, depth_intrin, pt0, pt1)
        print(f"✅ 每像素真实长度: {pixel_to_meter*1000:.6f} mm/pixel")

        # 显示图像
        cv2.imshow("Color Frame", color_image)
        cv2.waitKey(3000)  # 等待3000毫秒（3秒）后自动退出

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
