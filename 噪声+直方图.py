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

# 绘制图像和直方图
fig, axes = plt.subplots(4, 2, figsize=(12, 16))  # 4行，2列

images = [image, gaussian_noisy_image, uniform_noisy_image, salt_and_pepper_noisy_image]
titles = ['Original Image', 'Gaussian Noise', 'Uniform Noise', 'Salt and Pepper Noise']

for i, img in enumerate(images):
    # 显示图像
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(titles[i])
    axes[i, 0].axis('off')

    # 计算并显示直方图
    axes[i, 1].hist(img.ravel(), bins=256, range=[0, 256], color='black')
    axes[i, 1].set_title(titles[i] + " Histogram")
    axes[i, 1].set_xlim([0, 256])

plt.tight_layout()
plt.show()
