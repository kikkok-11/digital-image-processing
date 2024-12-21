import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('3.png')

# 转为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 绘制检测到的人脸区域
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 设置输出图像的固定大小（宽度x高度）
output_size = (800, 600)  # 比如将输出图像调整为600x400像素
resized_image = cv2.resize(image, output_size)

# 显示图像
cv2.imshow('Detected Faces', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
