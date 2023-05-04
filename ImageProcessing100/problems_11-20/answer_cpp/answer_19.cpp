/*
author: wenzy
modified date: 20230502
target: Implement the Sobel filter (3x3).
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

inline
double getGrayValue(Vec3b &pixel) {
  return 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];
}

Mat logFilter(Mat &I, Size K, double S) {
  int n_row = I.rows;
  int n_col = I.cols;
  int k_wid = K.width;
  int k_hei = K.height;
  double sum_kernel = 0.0;
  unique_ptr<double[]> kernel(new double [k_wid * k_hei]);
  unique_ptr<int[]> dx(new int [k_wid * k_hei]);
  unique_ptr<int[]> dy(new int [k_wid * k_hei]);
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  // 初始化 LoG kernel 和位移增量
  for (int i = 0; i < k_hei; i++)
      for (int j = 0; j < k_wid; j++) {
          int _idx = i * k_wid + j;
          dy[_idx] = -(k_wid>>1) + j;
          dx[_idx] = -(k_wid>>1) + i;
          double xys2 =  (double)dx[_idx] * dx[_idx] + (double)dy[_idx] * dy[_idx] - S * S;
          double _k = exp( -(xys2) / (2 * S * S) );
          _k *= (xys2) / (2 * CV_PI * pow(S, 6));
          kernel[_idx] = _k;
          sum_kernel += _k;
      }
  // 归一化
  for(int k = 0; k < 9; k++)
    kernel[k] /= sum_kernel;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        cout << 16.0 * kernel[i * 3 + j] << " ";
    }
    cout << endl;
  }

  // padding 滤波计算
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++) {
      // 注意计算过程使用浮点数
      double _pij = 0;
      for(int k = 0; k < 9; k++)
        _pij += getGrayValue(I.at<Vec3b>(i + dx[k], j + dy[k])) * kernel[k];
      _pij = fmax(_pij, 0);
      _pij = fmin(_pij, 255);
      // 最后再将结果转换为 uchar
      T.at<uchar>(i, j) = (uchar)_pij;
    }
  
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori_noise.jpg", IMREAD_COLOR);
  Mat A = logFilter(img, Size(3, 3), 1.6);
  imshow("before", img);
  imshow("logFilter", A);
  waitKey();
  return 0;
}