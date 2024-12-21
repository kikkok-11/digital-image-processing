import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2.jpg')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为浮动点数
gray = np.float32(gray)

# Harris 角点检测
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# 结果增强，便于可视化
dst = cv2.dilate(dst, None)

# 在原图上标记角点
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.show()
