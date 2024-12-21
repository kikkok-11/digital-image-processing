import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('2.jpg')


# RGB直方图均衡化
def rgb_histogram_equalization(image):
    r, g, b = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((r_eq, g_eq, b_eq))


# HSI分量转换
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


# HSI转RGB的函数
def hsi_to_rgb(hsi):
    h, s, i = hsi[:, :, 0], hsi[:, :, 1], hsi[:, :, 2]

    r = np.zeros(h.shape)
    g = np.zeros(h.shape)

    b = np.zeros(h.shape)

    for j in range(h.shape[0]):
        for k in range(h.shape[1]):
            H = h[j, k]
            S = s[j, k]
            I = i[j, k]

            if H < 120:
                b[j, k] = I * (1 - S)
                r[j, k] = I * (1 + (S * np.cos(H * np.pi / 180) / np.cos((60 - H) * np.pi / 180)))
                g[j, k] = 3 * I - (r[j, k] + b[j, k])
            elif H < 240:
                H -= 120
                r[j, k] = I * (1 - S)
                g[j, k] = I * (1 + (S * np.cos(H * np.pi / 180) / np.cos((60 - H) * np.pi / 180)))
                b[j, k] = 3 * I - (r[j, k] + g[j, k])
            else:
                H -= 240
                g[j, k] = I * (1 - S)
                b[j, k] = I * (1 + (S * np.cos(H * np.pi / 180) / np.cos((60 - H) * np.pi / 180)))
                r[j, k] = 3 * I - (g[j, k] + b[j, k])

    r = np.clip(r * 255, 0, 255).astype(np.uint8)
    g = np.clip(g * 255, 0, 255).astype(np.uint8)
    b = np.clip(b * 255, 0, 255).astype(np.uint8)

    return cv2.merge((b, g, r))


# HSI直方图均衡化
def hsi_histogram_equalization(image):
    hsi = rgb_to_hsi(image)
    h, s, i = cv2.split(hsi)

    # 对强度通道进行均衡化
    i_eq = cv2.equalizeHist((i * 255).astype(np.uint8)) / 255.0

    # 重新合并成HSI并转换回RGB
    hsi_eq = cv2.merge((h, s, i_eq))
    return hsi_to_rgb(hsi_eq)


# 进行均衡化
rgb_eq = rgb_histogram_equalization(image)
hsi_eq = hsi_histogram_equalization(image)

# 显示结果
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# RGB均衡化结果
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(rgb_eq, cv2.COLOR_BGR2RGB))
plt.title('RGB Histogram Equalization')
plt.axis('off')

# HSI均衡化结果
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(hsi_eq, cv2.COLOR_BGR2RGB))
plt.title('HSI Histogram Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()
