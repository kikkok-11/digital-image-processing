import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度模式
image_path = '2.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接以灰度模式读取

# 定义灰度级切片的阈值
thresholds = [0, 85, 170, 255]  # 根据需要调整阈值
segments = len(thresholds) - 1

# 创建一个新的数组用于存储切片结果
segmented_image = np.zeros_like(image)

# 创建变换函数
transform_map = np.arange(256, dtype=np.uint8)

# 根据灰度值进行切片，并构建变换函数
for i in range(segments):
    lower_bound = thresholds[i]
    upper_bound = thresholds[i + 1]
    mask = (image >= lower_bound) & (image < upper_bound)

    # 使用线性插值来映射值
    if i < segments - 1:  # 如果不是最后一个段
        new_value = (i + 1) * (255 // segments)
        transform_map[lower_bound:upper_bound] = new_value
    else:  # 对于最后一个段，直接赋值
        transform_map[lower_bound:upper_bound + 1] = 255

    segmented_image[mask] = transform_map[image[mask]]

# 显示原始图像和切片图像
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

# 绘制变换函数
plt.subplot(1, 3, 3)
plt.title("Transformation Function")
plt.plot(transform_map, color='blue')
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.xlabel("Original Gray Level")
plt.ylabel("Transformed Gray Level")
plt.grid()
plt.axhline(y=255, color='r', linestyle='--')  # 可选：最大值水平线
plt.axvline(x=thresholds[0], color='gray', linestyle='--')  # 可选：阈值线
plt.axvline(x=thresholds[1], color='gray', linestyle='--')  # 可选：阈值线
plt.axvline(x=thresholds[2], color='gray', linestyle='--')  # 可选：阈值线

plt.tight_layout()
plt.show()
