import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 定义Prewitt算子
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

# 使用滤波器计算梯度
gradient_x = cv2.filter2D(image, -1, prewitt_x)
gradient_y = cv2.filter2D(image, -1, prewitt_y)

# 计算边缘强度
edge_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# 转换为8位图像
gradient_x = np.uint8(np.clip(gradient_x, 0, 255))
gradient_y = np.uint8(np.clip(gradient_y, 0, 255))
edge_magnitude = np.uint8(np.clip(edge_magnitude, 0, 255))

# 显示原图、x方向、y方向和边缘检测结果
plt.figure(figsize=(12, 8))
'''
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
'''
plt.subplot(2, 2, 1)
plt.imshow(gradient_x, cmap='gray')
plt.title('X Direction Gradient')

plt.subplot(2, 2, 2)
plt.imshow(gradient_y, cmap='gray')
plt.title('Y Direction Gradient')

plt.subplot(2, 2, 3)
plt.imshow(edge_magnitude, cmap='gray')
plt.title('Prewitt Edge Detection')

plt.tight_layout()
plt.show()
