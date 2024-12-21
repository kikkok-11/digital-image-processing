import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image, noise_level=50):
    """添加椒盐噪声"""
    noisy_image = image.copy()
    noise = np.random.randint(0, noise_level, image.shape, dtype='uint8')
    noisy_image = cv2.add(noisy_image, noise)
    return noisy_image

# 读取原图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found or unable to load.")


# 设置滤波器内核大小
kernel_size = 15

# 均值滤波器
mean_filtered = cv2.blur(image, (kernel_size, kernel_size))

# 方框滤波器
box_filtered = cv2.boxFilter(image, -1, (kernel_size, kernel_size))

# 高斯滤波器
gaussian_filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 显示结果
plt.figure(figsize=(14, 10))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(mean_filtered, cmap='gray')
plt.title('Mean Filtered')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(box_filtered, cmap='gray')
plt.title('Box Filtered')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('Gaussian Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()
