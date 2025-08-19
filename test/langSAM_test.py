import cv2
import matplotlib.pyplot as plt
from lang_sam import LangSAM

# Step 1: 加载图像（支持jpg/png等）
image_path = "img/img_from_camera.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: 初始化模型
model = LangSAM()

# Step 3: 输入语言提示（你可以从LLM自动生成）
prompt = "the red apple on the right"

# Step 4: 执行预测
masks = model.predict(image_rgb, prompt)

# Step 5: 可视化分割结果
for i, mask in enumerate(masks):
    plt.figure()
    plt.imshow(image_rgb)
    plt.imshow(mask['segmentation'], alpha=0.5, cmap='jet')  # 用颜色叠加 mask
    plt.title(f"Mask {i+1}")
    plt.axis('off')
plt.show()
