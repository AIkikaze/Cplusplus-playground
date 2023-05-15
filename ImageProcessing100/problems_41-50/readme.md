<!--
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-10 08:30:17
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-15 13:51:13
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_41-50\readme.md
 * @Description: 
 * 
-->
# 问题 41-50

## 问题 41 Canny 边缘检测 第一步：边缘强度

问题 41-43 都主要是在介绍 Canny 边缘检测理论，其基本的算法思路为
1. 对图像进行高斯滤波
2. 在 x 方向和 y 方向上使用 Sobel 滤波，得到 x 和 y 方向上的一阶差分
3. 结合两方向上差分得到梯度的大小和方向，依据方向对梯度大小进行非极大值抑制，以使得边缘变得更细
4. 利用阈值筛选出理想的边缘图像

在问题 41 中，我们主要完成第一、二两步。这两步的实现思路是：
1. 首先对图像进行灰度化处理
2. 将图像进行高斯滤波 （核大小为 5x5，方差为 1.4）
3. 在 x 方向和 y 方向上使用 Sobel 滤波器，在此之上求出边缘梯度 $f_x$ 和 $f_y$
