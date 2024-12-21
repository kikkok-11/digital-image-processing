import cv2
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 对图像进行高斯平滑处理（Canny 算子依赖平滑）
smoothed_image = cv2.GaussianBlur(image, (5, 5), 1)

# 3. Canny 边缘检测
# 参数 100 和 200 分别为低阈值和高阈值，可根据图像调整
edges = cv2.Canny(smoothed_image, 100, 200)

# 4. 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(image, cmap="gray")
plt.subplot(1, 3, 2), plt.title("Smoothed Image"), plt.imshow(smoothed_image, cmap="gray")
plt.subplot(1, 3, 3), plt.title("Canny Edges"), plt.imshow(edges, cmap="gray")
plt.tight_layout()
plt.show()
