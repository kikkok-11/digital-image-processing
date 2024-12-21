import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Error: Image not found or unable to read.")

# 计算离散傅里叶变换
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# 将DFT结果移到频域中心
dft_shift = np.fft.fftshift(dft)

# 计算幅度谱
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = np.log(magnitude_spectrum + 1)  # 对数缩放

# 进行逆变换
dft_ishift = np.fft.ifftshift(dft_shift)  # 反移
img_back = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 获取幅度

# 显示结果
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# 频谱图
plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# 逆变换结果
plt.subplot(1, 3, 3)
plt.imshow(img_back, cmap='gray')
plt.title('Inverse DFT Result')
plt.axis('off')

plt.tight_layout()
plt.show()
