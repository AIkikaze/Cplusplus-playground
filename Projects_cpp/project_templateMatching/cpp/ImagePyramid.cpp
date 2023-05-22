#include "GeoMatch.hpp"
using namespace std;
using namespace cv;

void pyUpDown() {
  Mat src = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat dst, dst1;

	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
 
	//上采样（zoom in 放大）
	pyrUp(src, dst, Size(src.cols * 2, src.rows * 2));
	imshow("zoom in", dst);
	//降采样（zoom out 缩小）
	pyrDown(src, dst1, Size(src.cols / 2, src.rows / 2));
	imshow("zoom out", dst1);

  //高斯不同DOG（different of gaussian）
	//就是把同一张图像在不同的参数下做高斯模糊之后的结果相减，得到的输出图像;高斯不同是图像的内在特征，在灰度图像增强、角点检测中经常用到。
	Mat gray_src, dst2, dst3, dog_Image;
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	GaussianBlur(gray_src, dst2, Size(3, 3), 0, 0);
	imshow("dst2..", dst2);
	GaussianBlur(dst2, dst3, Size(3, 3), 0, 0);
	imshow("dst3..", dst3);
	subtract(dst2, dst3, dog_Image);
 
	//归一化显示
	normalize(dog_Image, dog_Image, 255, 0, NORM_MINMAX);
 
	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", dog_Image);

  waitKey();
}