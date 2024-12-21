import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('2.jpg')

# 计算HSI分量
def rgb_to_hsi(rgb):
    r, g, b = rgb[:, :, 0] / 255.0, rgb[:, :, 1] / 255.0, rgb[:, :, 2] / 255.0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(num / (den + 1e-10), -1, 1))

    h = np.zeros(r.shape)
    h[b <= g] = theta[b <= g]
    h[b > g] = 2 * np.pi - theta[b > g]
    h = h * (180.0 / np.pi)  # 转换为度数

    i = (r + g + b) / 3.0

    s = 1 - (3 / (r + g + b + 1e-10)) * np.minimum(np.minimum(r, g), b)

    hsi = np.zeros(rgb.shape)
    hsi[:, :, 0] = h
    hsi[:, :, 1] = s
    hsi[:, :, 2] = i

    return hsi


image_hsi = rgb_to_hsi(image)

# 分离RGB分量
r, g, b = cv2.split(image)

# 分离HSI分量
h, s, i = cv2.split(image_hsi)

# 显示结果
plt.figure(figsize=(12, 6))

# RGB分量图
plt.subplot(2, 4, 1)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')

# HSI分量图
plt.subplot(2, 4, 5)
plt.imshow(h, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(i, cmap='gray')
plt.title('Intensity Channel')
plt.axis('off')

plt.show()
