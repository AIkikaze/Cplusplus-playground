/*
author: wenzy
modified date: 20230502
target: implement the Gaussian filter (3 × 3, standard deviation 1.3) and 
remove the noise of imori_noise.jpg .
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

Mat gaussianFilter(Mat &I, Size S, double SIGMA) {
  int n_row = I.rows;
  int n_col = I.cols;
  int n_channel = I.channels();
  int n_wid = S.width;
  int n_hei = S.height;
  double kernel_sum = 0.0;
  unique_ptr<double[]> kernel(new double [n_wid * n_hei]);
  unique_ptr<int[]> dx(new int [n_wid * n_hei]);
  unique_ptr<int[]> dy(new int [n_wid * n_hei]);
  Mat T = Mat::zeros(n_row, n_col, CV_8UC3);

  // 初始化高斯滤波器核与位移增量
  for(int i = 0; i < n_hei; i++)
    for(int j = 0; j < n_wid; j++) {
      int _idx = i * n_wid + j;
      dy[_idx] = -(n_wid>>1) + j;
      dx[_idx] = -(n_hei>>1) + i;
      double xy2 = (double)dx[_idx] * dx[_idx] + (double)dy[_idx] * dy[_idx];
      kernel[_idx] = exp( - xy2 / (2 * SIGMA * SIGMA) ) / ( 2 * CV_PI * SIGMA * SIGMA );
      kernel_sum += kernel[_idx];
    }
  // 归一化处理
  for(int i = 0; i < n_wid * n_hei; i++)
    kernel[i] /= kernel_sum;

  // padding 滤波计算
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++)
      for(int c = 0; c < n_channel; c++) {
        double _p = 0.0;
        for(int _i = 0; _i < n_wid * n_hei; _i++) {
          if(i + dx[_i] < 0 || i + dx[_i] >= n_row || j + dy[_i] < 0 || j + dy[_i] >= n_col)
            continue;
          _p += kernel[_i] * I.at<Vec3b>(i + dx[_i], j + dy[_i])[c];
        }
        T.at<Vec3b>(i, j)[c] = (uchar)_p;
      }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori_noise.jpg", IMREAD_COLOR);
  Mat A = gaussianFilter(img, Size(3, 3), 1.3);
  imshow("before", img);
  imshow("gaussianFilter", A);
  waitKey();
  return 0;
}