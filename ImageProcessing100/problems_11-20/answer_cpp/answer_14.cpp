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

inline
double getGrayValue(Vec3b &pixel) {
  return 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];
}

Mat differentialFilter(Mat &I, int ORIENT[])
{
  int n_row = I.rows;
  int n_col = I.cols;
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  // padding 滤波计算
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++) {
        double _p0 = getGrayValue(I.at<Vec3b>(i, j));
        double _p1 = getGrayValue(I.at<Vec3b>(i + ORIENT[0], j + ORIENT[1]));
        T.at<uchar>(i, j) = abs(_p0 - _p1);
      }
  
  return T;
}

int main()
{
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  int orient[] = {-1, 0};
  Mat A = differentialFilter(img, orient);
  imshow("before", img);
  imshow("differentialFilter", A);
  waitKey();
  return 0;
}