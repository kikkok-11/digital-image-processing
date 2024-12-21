import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)


# 均值滤波器
kernel_size = 5
mean_filtered = cv2.blur(image, (kernel_size, kernel_size))

# 方框滤波器
box_filtered = cv2.boxFilter(image, -1, (kernel_size, kernel_size))

# 高斯滤波器
gaussian_filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 显示结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mean_filtered, cmap='gray')
plt.title('Mean Filtered')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(box_filtered, cmap='gray')
plt.title('Box Filtered')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('Gaussian Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()
