import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
image = cv2.imread('2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 使用 HoughLinesP 进行直线检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# 绘制检测到的直线段
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
