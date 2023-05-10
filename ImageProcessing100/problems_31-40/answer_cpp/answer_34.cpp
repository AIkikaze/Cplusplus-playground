/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-08 14:47:11
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 15:47:58
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_34.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

void preinit(Mat &I) {
  // 扩张原图的像素矩阵以便于后续计算
  int n_row = getOptimalDFTSize(I.rows);
  int n_col = getOptimalDFTSize(I.cols);
  copyMakeBorder(I, I, 0, n_row - I.rows, 0, n_col - I.cols, BORDER_CONSTANT, Scalar::all(0));
}

// dft: I(y, x) to G(k, l)
// G(k, l) = 1 / (rows * cols) \sum_y \sum_x I(y, x) e^{ - 2pi * j * (ky/rows + lx/cols) }
Mat myDft(const Mat &I) {
  complex<double> val;
  Mat T = Mat::zeros(I.size(), CV_64FC2);

  for(int k = 0; k < I.rows; k++) {
    for(int l = 0; l < I.cols; l++) {
      val = complex<double>(0.0, 0.0);
      for(int y = 0; y < I.rows; y++) {
        for(int x = 0; x < I.cols; x++) {
          double theta = -2.0 * CV_PI * ((double)k*y/I.rows + (double)l*x/I.cols);
          val += (double)I.at<uchar>(y, x) * complex<double>(cos(theta), sin(theta));
        }
      }
      val /= sqrt(I.rows * I.cols);
      T.at<Vec2d>(k, l)[0] = val.real();
      T.at<Vec2d>(k, l)[1] = val.imag();
    }
  }

  return T;
}

// idft: I(y, x) to G(k, l)
// I(y, x) = 1 / (rows * cols) \sum_y \sum_x G(k, l) e^{ 2pi * j * (ky/rows + lx/cols) }
Mat myIdft(const Mat &I) {
  complex<double> val;
  Mat T = Mat::zeros(I.size(), CV_8U);

  for(int y = 0; y < I.rows; y++) {
    for(int x = 0; x < I.cols; x++) {
      val = complex<double>(0.0, 0.0);
      for(int k = 0; k < I.rows; k++) {
        for(int l = 0; l < I.cols; l++) {
          complex<double> _val = complex<double>(I.at<Vec2d>(k, l)[0], I.at<Vec2d>(k, l)[1]);
          double theta = 2.0 * CV_PI * ((double)k*y/I.rows + (double)l*x/I.cols);
          val += _val * complex<double>(cos(theta), sin(theta));
        }
      }
      val /= sqrt(I.rows * I.cols);
      T.at<uchar>(y, x) = abs(val) > 255 ? 255 : fmax(abs(val), 0);
    }
  }

  return T;
}

void recenter(Mat &I) {
  // 将幅度谱剪裁为偶数行与偶数列(方便后面的重新排列）
	I = I(Range(0, I.rows & -2), Range(0, I.cols & -2));
  // 重新排列幅度谱的区域，使得幅度谱的原点位于图像中心
	int x0 = I.cols / 2;
	int y0 = I.rows / 2;
	Mat q0(I, Rect(0, 0, x0, y0));       //左上角图像
	Mat q1(I, Rect(x0, 0, x0, y0));      //右上角图像
	Mat q2(I, Rect(0, y0, x0, y0));      //左下角图像
	Mat q3(I, Rect(x0, y0, x0, y0));     //右下角图像
  // 交换第一象限和第三象限
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  // 交换第二象限和第四象限
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

void highpassFilter(Mat &I, double R) {
  // 中心化经过傅里叶变换的复数矩阵
  recenter(I);
  // 初始化高通滤波核
  double cy = (double)I.rows/2.0;
  double cx = (double)I.cols/2.0;
  double r_filter = sqrt(cy*cy + cx*cx) * R;
  for(int y = 0; y < I.rows; y++) {
    for(int x = 0; x < I.cols; x++) {
      double dist = sqrt(pow((double)y-cy, 2) + pow((double)x-cx, 2));
      if(dist < r_filter) {
        I.at<Vec2d>(y, x)[0] = 0.0;
        I.at<Vec2d>(y, x)[1] = 0.0;
      }
    }
  }
  // 再次中心化将矩阵还原
  recenter(I);
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_GRAYSCALE);
  
  preinit(img);
  Mat F = myDft(img);
  highpassFilter(F, 0.2);
  Mat iF = myIdft(F);

  imshow("origin", img);
  imshow("highpass", iF);
  waitKey();
  return 0;
}