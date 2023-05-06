# 问题 21-30

## 问题 21 直方图归一化 Histogram Normalization

有时直方图会偏向一边, 这时需要用到直方图归一化操作。

比如说，数据集中在 $0$ 处（左侧）的图像全体会偏暗，数据集中在 $255$ 处（右侧）的图像会偏亮。

如果直方图有所偏向，那么其[动态范围 (dynamic range)](https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%8C%83%E5%9B%B4)就会较低。为了使人能更清楚地看见图片，让直方图归一化、平坦化是十分必要的。

这种归一化直方图的操作被称作 **灰度变换 (Grayscale Transformation)**。像素点取值范围从  $[c,d]$  转换到  $[a,b]$  的过程由下式定义。这回我们将`imori_dark.jpg`的灰度扩展到  $[0, 255]$  范围：

$$
x_{out} =
\begin{cases}
  a & (\text{if}\quad x_{in}<c) \\
  \frac{b-a}{d-c}\ (x_{in}-c)+a&(\text{else if}\quad c\leq x_{in}<d)\\
  b&(\text{else})
\end{cases}
$$

## 问题 22 直方图操作 Histogram Operation

让直方图的平均值 $m_0=128$ ，标准差 $s_0=52$ 吧！

这里并不是变更直方图的动态范围，而是让直方图变得平坦。

可以使用下式将平均值为 $m$ 标准差为 $s$ 的直方图变成平均值为 $m_0$ 标准差为 $s_0$ 的直方图：

$$
x_{out}=\frac{s_0}{s}\  (x_{in}-m)+m_0
$$

## 问题 23 直方图均衡化 Histogram Equalization

来让均匀化直方图吧！直方图均衡化是使直方图变得平坦的操作，是不需要计算上面的问题中的平均值、标准差等数据使直方图的值变得均衡的操作。均衡化操作由以下式子定义。

$S$ 是总的像素数； $Z_{max}$ 是像素点的最大取值（在这里是 $255$ ）； $h(z)$ 表示取值为 $z$ 的累积分布函数：

$$
Z' = \frac{Z_{max}}{S} \  \sum\limits_{i=0}^z\ h(i)
$$

## 问题 24 伽玛校正 Gamma Correction [^1]

对`imori_gamma.jpg`进行伽马校正（ $c=1$ ， $g=2.2$ ）吧！

伽马校正用来对照相机等电子设备传感器的非线性光电转换特性进行校正。如果图像原样显示在显示器等上，画面就会显得很暗。伽马校正通过预先增大 RGB 的值来排除显示器的影响，达到对图像修正的目的。

由于下式引起非线性变换，其中 $c$ 是常数， $g$ 为伽马变量（通常取 $2.2$ ）。注意在下式中 $x$ 已经被归一化限定在 $[0,1]$ 范围内 ：

$$
x' = c\  {I_{in}}^ g
$$

因此，使用下面的式子进行伽马校正（并在矫正完成后将归一化的 $x$ 重新映回区间 $[0, 255]$）：

$$
I_{out} ={\frac{1}{c}\  I_{in}} ^ {\frac{1}{g}}
$$

## 问题 25 最邻近插值 Nearest-neighbor Interpolation

使用最邻近插值将图像放大 $1.5$ 倍吧！最近邻插值在图像放大时补充的像素取最临近的像素的值。由于方法简单，所以处理速度很快，但是放大图像画质劣化明显。

使用下面的公式放大图像吧！$I'$ 为放大后图像，$I$ 为放大前图像，$a$ 为放大率，方括号是四舍五入取整操作：

$$
I'(x,y) = I([\frac{x}{a}], [\frac{y}{a}])
$$

## 问题 26 双线性插值 Bilinear Interpolation

使用双线性插值将图像放大 $1.5$ 倍吧！

双线性插值考察 $4$ 邻域的像素点，并根据距离设置权值。虽然计算量增大使得处理时间变长，但是可以有效抑制画质劣化。

1. 放大后图像的座标 $(x',y')$ 除以放大率 $a$ ，可以得到对应原图像的座标 $(\lfloor \frac{x'}{a}\rfloor , \lfloor \frac{y'}{a}\rfloor)$ 。
2. 求原图像的座标 $(\lfloor \frac{x'}{a}\rfloor , \lfloor \frac{y'}{a}\rfloor)$ 周围 $4$ 邻域的座标 $I(x,y)$ ， $I(x+1,y)$ ， $I(x,y+1)$ ， $I(x+1, y+1)$ ：
3. 分别求这4个点与 $(\frac{x'}{a}, \frac{y'}{a})$ 的距离，根据距离设置权重： $w = \frac{d}{\sum\ d}$ 
4. 根据下式求得放大后图像 $(x',y')$ 处的像素值：

$$
\begin{gather*}
  d_x = \frac{x'}{a} - x\\
  d_y = \frac{y'}{a} - y\\
  I'(x',y') = (1-d_x)\  (1-d_y)\  I(x,y) + d_x\  (1-d_y)\  I(x+1,y) \\ 
  + (1-d_x)\  d_y\  I(x,y+1) + d_x\  d_y\  I(x+1,y+1)
\end{gather*}
$$

## 问题 27 双三次插值 Bicubic Interpolation

使用双三次插值将图像放大 $1.5$ 倍吧！

双三次插值是双线性插值的扩展，使用邻域 $16$ 像素进行插值。

各自像素间的距离由下式决定：

$$
\begin{gather*}
d_{x_1} = |\frac{x'}{a} - (x-1)|\quad 
d_{x_2} = |\frac{x'}{a}- x| \quad 
d_{x_3} = |\frac{x'}{a}- (x+1)|\quad 
d_{x_4} = |\frac{x'}{a} - (x+2)| \\
d_{y_1} = |\frac{y'}{a} - (y-1)|\quad 
d_{y_2} = |\frac{y'}{a} - y| \quad 
d_{y_3} = |\frac{y'}{a} - (y+1)| \quad 
d_{y_4} = |\frac{y'}{a} - (y+2)|
\end{gather*}
$$

权重由基于距离的函数取得。 $a$ 在大部分时候取 $-1$ 。大体上说，图中蓝色像素的距离 $|t|\leq 1$ ，绿色像素的距离 $1<|t|\leq 2$ ：

$$
h(t)=
\begin{cases}
(a+2)\ |t|^3 - (a+3)\ |t|^2 + 1 &\text{when}\quad |t|\leq 1  \\
a\ |t|^3 - 5\  a\ |t|^2 + 8\  a\  |t| - 4\  a&\text{when}\quad 1<|t|\leq 2\\
0&\text{else}
\end{cases}
$$

利用上面得到的权重，通过下面的式子扩大图像。将每个像素与权重的乘积之和除以权重的和。

$$
I'(x', y')=\frac{1}{\sum\limits_{j=1}^4\ \sum\limits_{i=1}^4\ h(d_{xi})\ h(d_{yj})}\ \sum\limits_{j=1}^4\ \sum\limits_{i=1}^4\ I(x+i-2,y+j-2)\ h(d_{xi})\ h(d_{yj})
$$

## 问题 28 仿射变换-平移 Afine Transformations 

利用仿射变换让图像在 $x$ 方向上 $+30$ ，在 $y$ 方向上 $-30$ 吧！仿射变换利用 $3\times3$ 的矩阵来进行图像变换。

变换的方式有平行移动（问题二十八）、放大缩小（问题二十九）、旋转（问题三十）、倾斜（问题三十一）等。

原图像记为 $(x,y)$ ，变换后的图像记为 $(x',y')$ 。

图像放大缩小矩阵为下式：
$$
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)=
\left(
\begin{matrix}
a&b\\
c&d
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)
$$
另一方面，平行移动按照下面的式子计算：
$$
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)=
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)+
\left(
\begin{matrix}
t_x\\
t_y
\end{matrix}
\right)
$$
把上面两个式子盘成一个：
$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
a&b&t_x\\
c&d&t_y\\
0&0&1
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$
但是在实际操作的过程中，如果一个一个地计算原图像的像素的话，处理后的像素可能没有在原图像中有对应的坐标。[^2]

[^2]: 这句话原文是“処理後の画像で値が割り当てられない可能性がでてきてしまう。”直译大概是”处理后的图像可能没有被分配到值。“我也不知道该怎么翻译才好……你们看输出图像左下角黑色的那一块，就是这种没有被”分配“到的情况。

因此，我们有必要对处理后的图像中各个像素进行仿射变换逆变换，取得变换后图像中的像素在原图像中的坐标。仿射变换的逆变换如下：
$$
\left(
\begin{matrix}
x\\
y
\end{matrix}
\right)=
\frac{1}{a\  d-b\  c}\ 
\left(
\begin{matrix}
d&-b\\
-c&a
\end{matrix}
\right)\  
\left(
\begin{matrix}
x'\\
y'
\end{matrix}
\right)-
\left(
\begin{matrix}
t_x\\
t_y
\end{matrix}
\right)
$$
这回的平行移动操作使用下面的式子计算。 $t_x$ 和 $t_y$ 是像素移动的距离。
$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
1&0&t_x\\
0&1&t_y\\
0&0&1
\end{matrix}
\right)\  
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$

## 问题 29 仿射变换-放大缩小 Afine Transformations 

1. 使用仿射变换，将图片在 $x$ 方向上放大 $1.3$ 倍，在 $y$ 方向上缩小至原来的 $\frac{4}{5}$ 。
2. 在上面的条件下，同时在 $x$ 方向上向右平移 $30$ （ $+30$ ），在 $y$ 方向上向上平移 $30$ （ $-30$ ）。

## 问题 30 仿射变换-旋转 Afine Transformations

1. 使用仿射变换，逆时针旋转 $30$ 度。
2. 使用仿射变换，逆时针旋转 $30$ 度并且能让全部图像显现（也就是说，单纯地做仿射变换会让图片边缘丢失，这一步中要让图像的边缘不丢失，需要耗费一些工夫）。

使用下面的式子进行逆时针方向旋转 $A$ 度的仿射变换：

$$
\left(
\begin{matrix}
x'\\
y'\\
1
\end{matrix}
\right)=
\left(
\begin{matrix}
\cos(A)&-\sin(A)&t_x\\
\sin(A)&\cos(A)&t_y\\
0&0&1
\end{matrix}
\right)\ 
\left(
\begin{matrix}
x\\
y\\
1
\end{matrix}
\right)
$$

[^1]: [这里](https://blog.csdn.net/candycat1992/article/details/46228771)是一篇写伽马校正比较好的文章，我觉得可以作为背景知识补充。