import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图片
image = cv2.imread('2.jpg')


# RGB均值滤波
def rgb_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))


# RGB拉普拉斯变换
def rgb_laplacian_transform(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # 将结果转换为uint8格式，进行归一化
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换为无符号8位整数
    return laplacian


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


# HSI均值滤波和拉普拉斯变换
def hsi_intensity_operations(image, kernel_size=3):
    hsi = rgb_to_hsi(image)
    h, s, i = cv2.split(hsi)

    # 在强度通道上进行均值滤波
    i_blurred = cv2.blur((i * 255).astype(np.uint8), (kernel_size, kernel_size)) / 255.0

    # 在强度通道上进行拉普拉斯变换
    i_laplacian = cv2.Laplacian((i * 255).astype(np.uint8), cv2.CV_64F)
    i_laplacian = cv2.convertScaleAbs(i_laplacian)  # 转换为无符号8位整数

    # 合并回HSI并转换回RGB
    hsi_filtered = cv2.merge((h, s, i_blurred))
    rgb_blurred = hsi_to_rgb(hsi_filtered)

    # 处理拉普拉斯变换的结果
    i_laplacian_normalized = i_laplacian / 255.0
    hsi_laplacian = cv2.merge((h, s, i_laplacian_normalized))
    rgb_laplacian = hsi_to_rgb(hsi_laplacian)

    return rgb_blurred, rgb_laplacian


# 进行操作
rgb_mean_result = rgb_mean_filter(image)
rgb_laplacian_result = rgb_laplacian_transform(image)
rgb_hsi_blurred, rgb_hsi_laplacian = hsi_intensity_operations(image)

# 显示结果
plt.figure(figsize=(12, 8))

# 原始图像
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# RGB均值滤波结果
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(rgb_mean_result, cv2.COLOR_BGR2RGB))
plt.title('RGB Mean Filter')
plt.axis('off')

# RGB拉普拉斯变换结果
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(rgb_laplacian_result, cv2.COLOR_BGR2RGB))
plt.title('RGB Laplacian Transform')
plt.axis('off')

# HSI强度均值滤波结果
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(rgb_hsi_blurred, cv2.COLOR_BGR2RGB))
plt.title('HSI Intensity Mean Filter')
plt.axis('off')

# HSI强度拉普拉斯变换结果
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(rgb_hsi_laplacian, cv2.COLOR_BGR2RGB))
plt.title('HSI Intensity Laplacian Transform')
plt.axis('off')

plt.tight_layout()
plt.show()
