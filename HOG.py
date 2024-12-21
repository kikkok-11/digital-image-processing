import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# 1. 加载图像
image = cv2.imread('2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 提取 HOG 特征
# HOG 参数：像素区块大小、单元格大小、块大小、以及方向数量
cell_size = (8, 8)
block_size = (2, 2)
nbins = 9

# 使用 skimage 的 HOG 提取函数来获取 HOG 特征和 HOG 图像
fd, hog_image = hog(image_gray, orientations=nbins, pixels_per_cell=cell_size,
                    cells_per_block=block_size, visualize=True)

# 3. 可视化 HOG 特征图
# HOG 特征图使用热图（heatmap）可视化
hog_image_rescaled = np.asarray(hog_image, dtype=np.uint8)

plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# HOG 特征图
plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Visualization')
plt.axis('off')

plt.show()

# 4. 画出归一化的 HOG 直方图
# 归一化后的 HOG 直方图
plt.figure(figsize=(6, 6))
plt.hist(fd, bins=nbins, color='gray', edgecolor='black')
plt.title('Normalized HOG Histogram')
plt.xlabel('Orientation')
plt.ylabel('Frequency')
plt.show()
