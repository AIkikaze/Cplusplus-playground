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

Mat embossFilter(Mat &I) {
  int n_row = I.rows;
  int n_col = I.cols;
  unique_ptr<int[]> kernel(new int [9]);
  unique_ptr<int[]> dx(new int [9]);
  unique_ptr<int[]> dy(new int [9]);
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  // 初始化 Emboss kernel 和位移增量
  int vec[] = {-2, -1, 0};
  for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
          int _idx = i * 3 + j;
          dy[_idx] = -1 + j;
          dx[_idx] = -1 + i;
          kernel[_idx] = vec[j] + i;
          if(!dx[_idx] && !dy[_idx]) 
            kernel[_idx]++;
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
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = embossFilter(img);
  imshow("before", img);
  imshow("embossFilter", A);
  waitKey();
  return 0;
}