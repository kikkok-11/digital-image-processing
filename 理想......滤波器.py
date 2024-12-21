import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Error: Image not found or unable to read.")

# 计算离散傅里叶变换
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 获取图像尺寸
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# 创建理想低通滤波器
def ideal_lowpass_filter(d, cutoff):
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, (1, 1), thickness=-1)
    return mask

# 创建巴特沃斯低通滤波器
def butterworth_lowpass_filter(d, cutoff, order=2):
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    return 1 / (1 + (d / cutoff) ** (2 * order))

# 创建高斯低通滤波器
def gaussian_lowpass_filter(d, cutoff):
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x, y = np.meshgrid(x, y)
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    return np.exp(-(d ** 2) / (2 * (cutoff ** 2)))

# 设置截止频率
cutoff = 30

# 应用滤波器
ideal_filter = ideal_lowpass_filter(dft_shift, cutoff)
butterworth_filter = butterworth_lowpass_filter(dft_shift, cutoff)
gaussian_filter = gaussian_lowpass_filter(dft_shift, cutoff)

# 过滤频谱
ideal_filtered = dft_shift * ideal_filter
butterworth_filtered = dft_shift * butterworth_filter[:, :, np.newaxis]
gaussian_filtered = dft_shift * gaussian_filter[:, :, np.newaxis]

# 逆变换
def inverse_dft(filtered_dft):
    dft_ishift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(dft_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

ideal_result = inverse_dft(ideal_filtered)
butterworth_result = inverse_dft(butterworth_filtered)
gaussian_result = inverse_dft(gaussian_filtered)

# 显示结果
plt.figure(figsize=(12, 8))

# 原图
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 理想滤波结果
plt.subplot(2, 2, 2)
plt.imshow(ideal_result, cmap='gray')
plt.title('Ideal Lowpass Filter Result')
plt.axis('off')

# 巴特沃斯滤波结果
plt.subplot(2, 2, 3)
plt.imshow(butterworth_result, cmap='gray')
plt.title('Butterworth Lowpass Filter Result')
plt.axis('off')

# 高斯滤波结果
plt.subplot(2, 2, 4)
plt.imshow(gaussian_result, cmap='gray')
plt.title('Gaussian Lowpass Filter Result')
plt.axis('off')

plt.tight_layout()
plt.show()
