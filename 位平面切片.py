import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 获取图像的高和宽
height, width = image.shape

# 准备存储每个位平面图像
bit_planes = []

# 进行位平面切片
for i in range(8):
    # 创建一个与原图像相同大小的图像，用于存储当前位平面
    bit_plane = np.zeros((height, width), dtype=np.uint8)

    # 将每个像素值的第i位提取出来
    bit_plane = (image >> i) & 1

    # 将其转换为 0 和 255 的图像（黑白）
    bit_plane = bit_plane * 255

    # 添加到结果列表
    bit_planes.append(bit_plane)

# 显示结果
plt.figure(figsize=(10, 10))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
