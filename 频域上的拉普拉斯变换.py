import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def laplacian_filter(shape):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    # 生成拉普拉斯滤波器
    h = -4 * np.ones((rows, cols))
    h[crow-1:crow+2, ccol-1:ccol+2] = 1  # 创建中心为1，周围为-1的矩阵
    return h

def apply_laplacian_filter(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = np.float32(img)

    # 进行离散傅里叶变换
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # 生成拉普拉斯滤波器
    laplacian_kernel = laplacian_filter(img.shape)

    # 应用拉普拉斯滤波器
    filtered_dft = dft_shifted * laplacian_kernel[:, :, np.newaxis]

    # 进行逆变换
    img_reconstructed = cv2.idft(np.fft.ifftshift(filtered_dft))
    img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])

    # 显示结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title('拉普拉斯增强图像')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数，输入图像路径
apply_laplacian_filter('2.jpg')