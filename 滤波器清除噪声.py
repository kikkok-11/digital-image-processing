import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载灰度图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 添加均匀噪声
def add_uniform_noise(image, low=-50, high=50):
    uniform_noise = np.random.uniform(low, high, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), uniform_noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, prob=0.02):
    noisy_image = image.copy()
    salt_pepper = np.random.rand(*image.shape)
    noisy_image[salt_pepper < prob / 2] = 0  # 盐噪声
    noisy_image[salt_pepper > 1 - prob / 2] = 255  # 椒噪声
    return noisy_image

# 获取每种噪声下的图像
gaussian_noisy_image = add_gaussian_noise(image)
uniform_noisy_image = add_uniform_noise(image)
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)

# 滤波操作
gaussian_denoised_image_gaussian = cv2.GaussianBlur(gaussian_noisy_image, (5, 5), 1.5)
uniform_denoised_image_gaussian = cv2.GaussianBlur(uniform_noisy_image, (5, 5), 1.5)
salt_and_pepper_denoised_image_median = cv2.medianBlur(salt_and_pepper_noisy_image, 5)

# 显示对比图
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# 设置标题
titles = [
    'Original Image', 'Gaussian Noise', 'Denoised (Gaussian Filter)', 'Histogram',
    'Original Image', 'Uniform Noise', 'Denoised (Gaussian Filter)', 'Histogram',
    'Original Image', 'Salt and Pepper Noise', 'Denoised (Median Filter)', 'Histogram'
]

# 高斯噪声前后对比
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title(titles[0])
axes[0, 1].imshow(gaussian_noisy_image, cmap='gray')
axes[0, 1].set_title(titles[1])
axes[0, 2].imshow(gaussian_denoised_image_gaussian, cmap='gray')
axes[0, 2].set_title(titles[2])

# 均匀噪声前后对比
axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title(titles[4])
axes[1, 1].imshow(uniform_noisy_image, cmap='gray')
axes[1, 1].set_title(titles[5])
axes[1, 2].imshow(uniform_denoised_image_gaussian, cmap='gray')
axes[1, 2].set_title(titles[6])

# 椒盐噪声前后对比
axes[2, 0].imshow(image, cmap='gray')
axes[2, 0].set_title(titles[8])
axes[2, 1].imshow(salt_and_pepper_noisy_image, cmap='gray')
axes[2, 1].set_title(titles[9])
axes[2, 2].imshow(salt_and_pepper_denoised_image_median, cmap='gray')
axes[2, 2].set_title(titles[10])


# 关闭坐标轴并展示
for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
