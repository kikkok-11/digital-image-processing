import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 计算原图的直方图
hist_original, bins = np.histogram(image.flatten(), 256, [0, 256])

# 绘制原图的直方图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.bar(bins[:-1], hist_original, width=1, color='black')

# 进行直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 计算均衡化后的直方图
hist_equalized, bins = np.histogram(equalized_image.flatten(), 256, [0, 256])

# 绘制均衡化后的直方图
plt.subplot(1, 2, 2)
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.bar(bins[:-1], hist_equalized, width=1, color='black')

# 显示原图和均衡化后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()
