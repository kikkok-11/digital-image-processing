import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 定义 Prewitt 水平和垂直算子
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
prewitt_y = np.array([[ 1,  1,  1],
                      [ 0,  0,  0],
                      [-1, -1, -1]])

# 3. 进行卷积操作，计算水平和垂直梯度
gradient_x = cv2.filter2D(image, -1, prewitt_x)
gradient_y = cv2.filter2D(image, -1, prewitt_y)

# 4. 计算梯度幅值
gradient_magnitude = cv2.magnitude(gradient_x.astype(np.float32), gradient_y.astype(np.float32))

# 5. 阈值化处理
_, thresholded = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

# 6. 显示结果
plt.figure(figsize=(10, 6))
##plt.subplot(2, 3, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray")
plt.subplot(2, 2, 1), plt.title("Prewitt X"), plt.imshow(gradient_x, cmap="gray")
plt.subplot(2, 2, 2), plt.title("Prewitt Y"), plt.imshow(gradient_y, cmap="gray")
plt.subplot(2, 2, 3), plt.title("Gradient Magnitude"), plt.imshow(gradient_magnitude, cmap="gray")
plt.subplot(2, 2, 4), plt.title("Thresholded"), plt.imshow(thresholded, cmap="gray")
plt.tight_layout()
plt.show()
