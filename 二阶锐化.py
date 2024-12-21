import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取并转换图像为灰度图
image = cv2.imread('2.jpg')  # 替换为你的图片路径
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 定义拉普拉斯算子（4邻域或8邻域都可以选择）
laplacian_operator_4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
laplacian_operator_8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

# 使用OpenCV中的Laplacian函数进行二阶锐化
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# 绝对值取整并转换为8位无符号整数
laplacian = np.uint8(np.absolute(laplacian))

# 显示原图、算子以及锐化结果
plt.figure(figsize=(12, 6))

# 原始灰度图
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
'''
# 拉普拉斯算子（4邻域）
plt.subplot(2, 3, 2)
plt.title('Laplacian Operator (4-neighborhood)')
plt.imshow(laplacian_operator_4, cmap='gray', interpolation='none')

# 拉普拉斯算子（8邻域）
plt.subplot(2, 3, 3)
plt.title('Laplacian Operator (8-neighborhood)')
plt.imshow(laplacian_operator_8, cmap='gray', interpolation='none')
'''
# 二阶锐化结果（拉普拉斯边缘检测）
plt.subplot(2, 3, 2)
plt.title('Laplacian Edge Detection')
plt.imshow(laplacian, cmap='gray')

plt.tight_layout()
plt.show()
