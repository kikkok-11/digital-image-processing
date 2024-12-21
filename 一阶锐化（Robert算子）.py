import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取并转换图像为灰度图
image = cv2.imread('2.jpg')  # 替换为图片路径
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 定义 Roberts 算子的滤波器
roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float32)  # 垂直方向
roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # 水平方向

# 使用卷积操作计算梯度
vertical = cv2.filter2D(gray_image, -1, roberts_cross_v)
horizontal = cv2.filter2D(gray_image, -1, roberts_cross_h)

# 计算梯度的幅值
gradient_magnitude = np.sqrt(np.square(horizontal) + np.square(vertical))

# 将梯度幅值归一化
gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

# 显示原始图像、算子滤波器和结果
plt.figure(figsize=(15, 5))

# 显示原始灰度图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')

# 显示 Roberts 边缘检测结果
plt.subplot(1, 2, 2)
plt.title('Roberts Edge Detection')
plt.imshow(gradient_magnitude, cmap='gray')

plt.show()
