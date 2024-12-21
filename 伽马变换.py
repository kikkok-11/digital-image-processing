import cv2 as cv
import copy
import numpy as np
import matplotlib.pyplot as plt

# 读入原始图像
img = cv.imread('2.jpg', 1)

# 灰度化处理
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 伽马变换
gamma = copy.deepcopy(gray)
rows, cols = gray.shape
for i in range(rows):
    for j in range(cols):
        gamma[i][j] = 3 * pow(gamma[i][j], 0.8)

# 绘制伽马变换函数
def gamma_transformation_function(gamma_value, c=3):
    x = np.linspace(0, 255, 256)  # 输入像素值范围
    y = c * np.power(x, gamma_value)  # 伽马变换公式
    return x, y

# 绘制转换函数
x, y = gamma_transformation_function(0.8)

plt.figure()
plt.plot(x, y, label="Gamma=0.8")
plt.title("Gamma Transformation Function")
plt.xlabel("Input pixel value")
plt.ylabel("Output pixel value")
plt.legend()
plt.grid(True)
plt.show()

# 显示图像
cv.imshow('Original Image', img)
cv.imshow('Gray Image', gray)
cv.imshow('Gamma Corrected Image', gamma)

# 等待按键关闭窗口
cv.waitKey(0)
cv.destroyAllWindows()
