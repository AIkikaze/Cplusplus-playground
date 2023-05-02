/*
author: wenzy
modified date: 20230502
remove the noise of imori_noise.jpg .
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

Mat max_minFilter(Mat &I, Size S) {
  int n_row = I.rows;
  int n_col = I.cols;
  int n_channel = I.channels();
  int n_wid = S.width;
  int n_hei = S.height;
  unique_ptr<int[]> dx(new int [n_wid * n_hei]);
  unique_ptr<int[]> dy(new int [n_wid * n_hei]);
  Mat T = Mat::zeros(n_row, n_col, CV_8UC3);

  // 初始化 motion 滤波器核与位移增量
  for(int i = 0; i < n_hei; i++)
    for(int j = 0; j < n_wid; j++) {
      int _idx = i * n_wid + j;
      dy[_idx] = -(n_wid>>1) + j;
      dx[_idx] = -(n_hei>>1) + i;
    }
  
  // padding 滤波计算
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++)
      for(int c = 0; c < n_channel; c++) {
        uchar _p_max = 0;
        uchar _p_min = 255;
        for(int _i = 0; _i < n_wid * n_hei; _i++) {
          if(i + dx[_i] < 0 || i + dx[_i] >= n_row || j + dy[_i] < 0 || j + dy[_i] >= n_col) {
            _p_min = 0;
            continue;
          }
          if(I.at<Vec3b>(i + dx[_i], j + dy[_i])[c] > _p_max)
            _p_max = I.at<Vec3b>(i + dx[_i], j + dy[_i])[c];
          if(I.at<Vec3b>(i + dx[_i], j + dy[_i])[c] < _p_min)
            _p_min = I.at<Vec3b>(i + dx[_i], j + dy[_i])[c];
        }
        T.at<Vec3b>(i, j)[c] = _p_max - _p_min;
      }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = max_minFilter(img, Size(3, 3));
  imshow("before", img);
  imshow("max_minFilter", A);
  waitKey();
  return 0;
}