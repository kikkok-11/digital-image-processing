import cv2 as cv
img = cv.imread('1.jpg',1)
img_1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('gray',img_1)
cv.imshow('colour',img)
cv.waitKey(0)
