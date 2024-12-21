import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

### 1. 计算拉普拉斯绝对值，获取边缘图像
laplacian = cv2.Laplacian(image, cv2.CV_64F)  # 计算拉普拉斯梯度
laplacian_abs = np.abs(laplacian)  # 取绝对值
laplacian_abs_normalized = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

### 2. 指定一个阈值 T
T = 50  # 可根据实际图像情况调整阈值

### 3. 阈值处理，生成二值模板图像
_, gT = cv2.threshold(laplacian_abs_normalized, T, 255, cv2.THRESH_BINARY)

# 将二值模板图像用于生成强边缘模板
template_mask = gT.astype(bool)

### 4. 仅用模板位置的像素计算直方图
# 筛选出 f(x, y) 中对应 gT(x, y) 为1的位置像素
selected_pixels = image[template_mask]  # 从原图中提取强边缘位置的像素

# 计算这些像素的直方图
hist = cv2.calcHist([selected_pixels], [0], None, [256], [0, 256])
hist_norm = hist / hist.sum()

### 5. 使用步骤4中的直方图全局分割 f(x, y)
# 计算基于直方图的全局阈值
total_pixels = selected_pixels.size
threshold = 0
max_between_class_variance = 0

for t in range(256):
    # 直方图分为两类
    w0 = sum(hist[:t]) / total_pixels  # 前景（低灰度值）权重
    w1 = sum(hist[t:]) / total_pixels  # 背景（高灰度值）权重

    if w0 == 0 or w1 == 0:  # 跳过不合理分割
        continue

    # 计算两类的均值
    mu0 = sum([i * hist[i] for i in range(t)]) / sum(hist[:t])
    mu1 = sum([i * hist[i] for i in range(t, 256)]) / sum(hist[t:])

    # 类间方差
    between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

    # 更新最大类间方差及对应阈值
    if between_class_variance > max_between_class_variance:
        max_between_class_variance = between_class_variance
        threshold = t

# 根据计算的全局阈值对原始图像进行分割
_, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

print(f"基于强边缘直方图计算的全局阈值: {threshold}")

### 显示处理过程与结果
plt.figure(figsize=(15, 10))
'''
# 原始图像
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
'''
# 拉普拉斯绝对值图像
plt.subplot(2, 2, 1)
plt.title("Laplacian Absolute Value")
plt.imshow(laplacian_abs_normalized, cmap='gray')
plt.axis('off')

# 二值模板图像 gT(x, y)
plt.subplot(2, 2, 2)
plt.title(f"Thresholded Template (T={T})")
plt.imshow(gT, cmap='gray')
plt.axis('off')

# 强边缘像素直方图
plt.subplot(2, 2, 3)
plt.title("Selected Pixels Histogram")
plt.plot(hist_norm, color='black')
plt.xlabel("Pixel Intensity")
plt.ylabel("Normalized Frequency")

# 基于直方图的分割结果
plt.subplot(2, 2, 4)
plt.title("Segmented Image (Custom Threshold)")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
