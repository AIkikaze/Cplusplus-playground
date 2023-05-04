# 问题 01-10

## 问题 01 通道交换 Channel Swap

读取图像，将 RGB 通道的图片转换为 BRG 通道（注意当 `imread` 的常量参数设置为 `IMREAD_COLOR` 时，opencv 的像素矩阵按 BRG 的顺序存储。

## 问题 02 灰度化 Grayscale

将图像灰度化吧！

灰度是一种图像亮度的表示方法，通过下式计算：

$$
Y = 0.2126\  R + 0.7152\  G + 0.0722\  B
$$

## 问题 03 二值化 Thresholding

把图像进行二值化吧。

二值化是将图像使用黑和白两种颜色表示的方法。

我们将灰度的阈值设置为 $128$ 来进行二值化，即：

$$
y=
\begin{cases}
0& (\text{if}\quad y < 128) \\
255& (\text{else})
\end{cases}
$$

## 问题 04 大津算法 Ostu's method

大津二值化法用来确定灰度阈值以对图像进行二值化，这一方法假定图像根据双模直方图（前景像素和背景像素）把包含两类像素，于是它要计算能将两类分开的最佳阈值，使得它们的**类内方差**最小。由于两两平方距离恒定，所以即它们的**类间方差**最大。

注：
1. 大津二值化法粗略的（有一些不同）来说就是一维费舍尔判别分析(类似LDA)的离散化模拟。
2. 原始方法的多级阈值扩展称为多大津算法。[^3]

我们记 $\sigma_{intra}^2, \sigma_{inter}^2$ 为类内方差和类间方差，并将这两种方差定义为( $k \in \{1, 2\}$ )：

$$ 
\begin{gather*}
s_{intra}^2 = \sum_{k} w_k \sum_{class(i) = k} p(i)(i - \mu_k)^2 = \sum_{k} w_k \sigma_k^2 \\
s_{inter}^2 = \sum_{k}w_k (\mu_k - \mu_{tol})^2
\end{gather*}
$$ 

特别地，对于确定二值化阈值的问题，分类器  $class(i)$  可以写成

$$ 
class(i, T) =   
  \begin{cases}
  & 0, \ 0 \leq i < T \\ 
  & 1, \ T \leq i < 256 
  \end{cases} 
$$ 

我们的目标是找到最合适的二值化阈值 $T^*$ ，使得下式尽可能小：

$$  
\begin{gather}
T^* = argmax_{T} \ \frac{s_{inter}^2}{s_{intra}^2}
\end{gather} 
$$ 

在简化问题时，记灰度图像中各个像素的灰度有离散概率分布为  $p(i)$  , 满足  $\sum_{i} p(i) = 1$  ，则我们能够使用的若干关系如下

$$ 
\begin{gather}
\mu_k = \frac{\sum_{class(i) = k} i p(i)}{\sum_{class(i) = k} p(i)} \\
\sigma_k^2 = \sum_{class(i) = k} p(i) [i - \mu_k]^2 = E[(i - \mu_k)^2] = E[i^2] - \mu_k^2 \\
w_k = \sum_{class(i) = k} p(i) \\
\mu_{tol} = \sum_{i} i p(i) = \sum_{k} \sum_{class(i) = k} ip(i) = \sum_k w_k \mu_k \\
\sigma_{tol}^2 = \sum_{i} p(i) [i - \mu_{tol}]^2 = s_{inter}^2 + s_{intra}^2 \\
\end{gather}
$$ 

其中最后一式需要计算检验。由于  $\sigma^2$  为定植，故原问题(1)可以等价于求最小的类内方差，或最大的类间方差。

另外需要注意的是，Ostu方法和fihser线性判别有系数上的不同：

一维二分类情形下由fisher判别式所确定的阈值满足

$$ 
\begin{gather}
T^*_{fisher} = \argmax_{T} \  \frac{(\mu_0 - \mu_1)^2}{\sigma_0^2 + \sigma_1^2}
\end{gather}
$$ 

而 Ostu方法确定的阈值满足（其中需要将(5)式代入  $s_{inter}^2$  中展开计算）

$$ 
\begin{gather}
T^*_{ostu} = \argmax_{T} \  \frac{w_0w_1(\mu_0 - \mu_1)^2}{w_0\sigma_0^2 + w_1\sigma_1^2} 
\end{gather}
$$ 

下面，我们简述算法实现的流程：

```
# Ostu's method
1. 计算每个灰度级的直方图和概率
3. 遍历所有可能的灰度级 T
  3.1. 计算分类比 w_k(T) 和类均值 mu_k(T)
  3.2. 计算类间方差 s2_inter
  3.3. 更新最大的类间方差 smax_inter 并记录第一次取到最大方差的灰度值 Tstar_l
  3.4. 记录最后一次取到最大方差的灰度值 Tstar_r
4. 对两次记录下的灰度值 Tstar_l 和 Tstar_r 取平均值即为理想的二值化阈值 Tstar
```

## 问题 05 色彩空间变换 BGR-HSV

将使用 $\text{HSV}$ 表示色彩的图像的色相反转吧！

 $\text{HSV}$ 即使用**色相（Hue）、饱和度（Saturation）、明度（Value）**来表示色彩的一种方式。

* 色相：将颜色使用 $0^{\circ}$ 到 $360^{\circ}$ 表示，就是平常所说的颜色名称，如红色、蓝色。色相与数值按下表对应：

| 红          | 黄           | 绿            | 青色          | 蓝色          | 品红          | 红            |
| ----------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- |
| $0^{\circ}$ | $60^{\circ}$ | $120^{\circ}$ | $180^{\circ}$ | $240^{\circ}$ | $300^{\circ}$ | $360^{\circ}$ |

* 饱和度：是指色彩的纯度，饱和度越低则颜色越黯淡（ $0\leq S < 1$ ）；
* 明度：即颜色的明暗程度。数值越高越接近白色，数值越低越接近黑色（ $0\leq V < 1$ ）；

从 $\text{RGB}$ 色彩表示转换到 $\text{HSV}$ 色彩表示通过以下方式计算： $\text{RGB}$ 的取值范围为 $[0, 255]$ ，令：

$$ 
\text{Max}=\max(R, G, B)\\
\text{Min}=\min(R, G, B)
$$ 

色相：

$$ 
H=\begin{cases}
0&(\text{if}\ \text{Min}=\text{Max})\\
60\  \frac{G-R}{\text{Max}-\text{Min}}+60&(\text{if}\ \text{Min}=B)\\
60\  \frac{B-G}{\text{Max}-\text{Min}}+180&(\text{if}\ \text{Min}=R)\\
60\  \frac{R-B}{\text{Max}-\text{Min}}+300&(\text{if}\ \text{Min}=G)
\end{cases}
$$ 

饱和度：

$$ 
S=\text{Max}-\text{Min}
$$ 

明度：

$$ 
V=\text{Max}
$$ 

从 $\text{HSV}$ 色彩表示转换到 $\text{RGB}$ 色彩表示通过以下方式计算：

$$ 
C = S\\
H' = \frac{H}{60}\\
X = C\  (1 - |H' \mod 2 - 1|)\\
(R, G, B)=(V-C)\ (1, 1, 1)+\begin{cases}
(0, 0, 0)&  (\text{if H is undefined})\\
(C, X, 0)&  (\text{if}\quad 0 \leq H' < 1)\\
(X, C, 0)&  (\text{if}\quad 1 \leq H' < 2)\\
(0, C, X)&  (\text{if}\quad 2 \leq H' < 3)\\
(0, X, C)&  (\text{if}\quad 3 \leq H' < 4)\\
(X, 0, C)&  (\text{if}\quad 4 \leq H' < 5)\\
(C, 0, X)&  (\text{if}\quad 5 \leq H' < 6)
\end{cases}
$$ 

请将色相反转（色相值加 $180$ ），然后再用 $\text{RGB}$ 色彩空间表示图片。

## 问题 06 减色处理

我们将图像的值由 $256^3$ 压缩至 $4^3$ ，即将 $\text{RGB}$ 的值只取 $\{32, 96, 160, 224\}$ 。这被称作色彩量化。色彩的值按照下面的方式定义：

$$
\text{val}=
\begin{cases}
  32 & (0 \leq \text{var} <  64)\\
  96 & (64\leq \text{var} < 128)\\
  160 &(128\leq \text{var} < 192)\\
  224 &(192\leq \text{var} < 256)
\end{cases}
$$

## 问题 07 平均池化 Average Pooling

将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值。

我们将这种把图片使用均等大小网格分割，并求网格内代表值的操作称为**池化（Pooling）**。

池化操作是**卷积神经网络（Convolutional Neural Network）**中重要的图像处理方式。平均池化按照下式定义：

$$
v=\frac{1}{|R|}\  \sum\limits_{i=1}^R\ v_i
$$

请把大小为 $128\times128$ 的 `imori.jpg` 使用 $8\times8$ 的网格做平均池化。

## 问题 08 最大池化 Max Pooling

网格内的值不取平均值，而是取网格内的最大值进行池化操作。

## 问题 09 高斯滤波

使用高斯滤波器（ $3\times3$ 大小，标准差 $\sigma=1.3$ ）来对 `imori_noise.jpg` 进行降噪处理吧！

高斯滤波器是一种可以使图像**平滑**的滤波器，用于去除**噪声**。可用于去除噪声的滤波器还有中值滤波器（参见问题十），平滑滤波器（参见问题十一）、LoG滤波器（参见问题十九）。

高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为**卷积核（kernel）**或者**滤波器（filter）**。

但是，由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补 $0$ 。这种方法称作**Zero Padding**。并且权值 $g$ （卷积核）要进行[归一化操作](https://blog.csdn.net/lz0499/article/details/54015150)（ $\sum\ g = 1$ ）。

按下面的高斯分布公式计算权值：

$$ 
g(x, y, \sigma)=\frac{1}{2\  \pi\ \sigma^2}\  e^{-\frac{x^2+y^2}{2\  \sigma^2}}
$$ 

标准差 $\sigma=1.3$ 的 $8-$ 近邻的常用高斯滤波器如下：

$$ 
K=\frac{1}{16}  
  \begin{bmatrix}
   1 & 2 & 1 \\
   2 & 4 & 2 \\
   1 & 2 & 1
  \end{bmatrix}
$$ 

实际计算得到的高斯滤波器如下：

$$ 
K=\frac{1}{16}   
  \begin{bmatrix}
   1.43059 & 1.92311 & 1.43059 \\
   1.92311 & 2.5852 & 1.92311 \\
   1.43059 & 1.92311 & 1.43059
  \end{bmatrix}
$$ 

## 问题 10 中值滤波 Median Filter

使用中值滤波器（ $3\times3$ 大小）来对 `imori_noise.jpg` 进行降噪处理吧！

中值滤波器是一种可以使图像平滑的滤波器。这种滤波器用滤波器范围内（在这里是 $3\times3$ ）像素点的中值进行滤波，请在这里也采用Zero Padding。

[^3]: Ping-Sung Liao and Kse-Sheng Chen and Pau-Choo Chung. A Fask Algorikhm for Mulkilevel Khresholding. J. Inf. Sci. Eng. 2001, 17 (5): 713–727.
