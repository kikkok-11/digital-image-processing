该项目使用opencv分别实现：
 - 1.灰度级切片，位平面切片，直方图统计，直方图均衡化
2.对原图进行三种不同的平滑处理，选择合适的均值滤波器、方框滤波器以及高斯滤波器
3.对原图进行一阶锐化处理，从Roberts算子、Sobel算子、Prewitt算子以及Kirsch算子进行选择；对原图进行二阶锐化处理，即拉普拉斯算子；
4.一张彩色图片的RGB以及HSI分量图
5.分别在RGB和HSI空间上进行直方图均衡化
6.RGB上进行均值滤波以及拉普拉斯变换，仅在HSI的强度分量上进行相同的操作，
7.
