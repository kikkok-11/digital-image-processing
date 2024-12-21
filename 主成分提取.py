import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取图片并转换为灰度图
img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 选择较小的主成分数
n_components = min(50, img.shape[0])  # 确保n_components不超过图像的高度

# 对每一行进行PCA
pca = PCA(n_components=n_components)
img_pca = pca.fit_transform(img)

# 将提取的主成分重新构建为图像
img_pca_reconstructed = pca.inverse_transform(img_pca)
img_pca_reconstructed = img_pca_reconstructed.reshape(img.shape)

# 恢复图像
img_reconstructed = pca.inverse_transform(img_pca)
img_reconstructed = img_reconstructed.reshape(img.shape)

# 显示原始图像、主成分图像与恢复后的图像
plt.figure(figsize=(15, 5))

# 显示原始图像
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

# 显示提取主成分后的图像
plt.subplot(1, 3, 2)
plt.title("PCA Extracted Image")
plt.imshow(img_pca_reconstructed, cmap='gray')

# 显示恢复后的图像
plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(img_reconstructed, cmap='gray')

plt.show()
