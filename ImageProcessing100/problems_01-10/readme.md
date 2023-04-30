# 问题 01-10

# 问题 03 大津算法 Ostu's method

大津二值化法用来确定灰度阈值以对图像进行二值化，这一方法假定图像根据双模直方图（前景像素和背景像素）把包含两类像素，于是它要计算能将两类分开的最佳阈值，使得它们的**类内方差**最小。由于两两平方距离恒定，所以即它们的**类间方差**最大。

注：
1. 大津二值化法粗略的（有一些不同）来说就是一维费舍尔判别分析(类似LDA)的离散化模拟。
2. 原始方法的多级阈值扩展称为多大津算法。[^3]

我们记 $\sigma_{intra}^2, \sigma_{inter}^2$ 为类内方差和类间方差，并将这两种方差定义为($k \in \{1, 2\}$)：

$$
\begin{gather*}
s_{intra}^2 = \sum_{k} w_k \sum_{class(i) = k} p(i)(i - \mu_k)^2 = \sum_{k} w_k \sigma_k^2 \\
s_{inter}^2 = \sum_{k}w_k (\mu_k - \mu_{tol})^2
\end{gather*}
$$

特别地，对于确定二值化阈值的问题，分类器 $class(i)$ 可以写成

$$
class(i, T) = \left\{
\begin{align*}
 & 0, \ 0 \leq i < T \\ 
 & 1, \ T \leq i < 256 
\end{align*}
\right.
$$

我们的目标是找到最合适的二值化阈值$T^*$，使得下式尽可能小：

$$ 
\begin{gather}
T^* = argmax_{T} \ \frac{s_{inter}^2}{s_{intra}^2}
\end{gather} 
$$

在简化问题时，记灰度图像中各个像素的灰度有离散概率分布为 $p(i)$ , 满足 $\sum_{i} p(i) = 1$ ，则我们能够使用的若干关系如下

$$
\begin{gather}
\mu_k = \frac{\sum_{class(i) = k} i p(i)}{\sum_{class(i) = k} p(i)} \\
\sigma_k^2 = \sum_{class(i) = k} p(i) [i - \mu_k]^2 = E[(i - \mu_k)^2] = E[i^2] - \mu_k^2 \\
w_k = \sum_{class(i) = k} p(i) \\
\mu_{tol} = \sum_{i} i p(i) = \sum_{k} \sum_{class(i) = k} ip(i) = \sum_k w_k \mu_k \\
\sigma_{tol}^2 = \sum_{i} p(i) [i - \mu_{tol}]^2 = s_{inter}^2 + s_{intra}^2 \\
\end{gather}
$$

其中最后一式需要计算检验。由于 $\sigma^2$ 为定植，故原问题(1)可以等价于求最小的类内方差，或最大的类间方差。

另外需要注意的是，Ostu方法和fihser线性判别有系数上的不同：

一维二分类情形下由fisher判别式所确定的阈值满足

$$
T^*_{fisher} = argmax_{T} \ \frac{(\mu_0 - \mu_1)^2}{\sigma_0^2 + \sigma_1^2}
$$

而 Ostu方法确定的阈值满足（其中需要将(5)式代入 $s_{inter}^2$ 中展开计算）

$$
T^*_{ostu} = argmax_{T} \ \frac{w_0w_1(\mu_0 - \mu_1)^2}{w_0\sigma_0^2 + w_1\sigma_1^2} 
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

[^3]: Ping-Sung Liao and Kse-Sheng Chen and Pau-Choo Chung. A Fask Algorikhm for Mulkilevel Khresholding. J. Inf. Sci. Eng. 2001, 17 (5): 713–727.