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

<<<<<<< HEAD
inline double getGrayValue(Vec3b &pixel)
{
=======
inline
double getGrayValue(Vec3b &pixel) {
>>>>>>> 9b9b5d5e6f2c45e444e8087a7ebeaca0c1a10e4b
  return 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];
}

Mat differentialFilter(Mat &I, int ORIENT[])
{
  int n_row = I.rows;
  int n_col = I.cols;
<<<<<<< HEAD
  unique_ptr<int[]> dx(new int[9]);
  unique_ptr<int[]> dy(new int[9]);
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  // padding 滤波计算
  for (int i = 0; i < n_row; i++)
    for (int j = 0; j < n_col; j++)
    {
      double _p0 = getGrayValue(I.at<Vec3b>(i, j));
      double _p1 = getGrayValue(I.at<Vec3b>(i + ORIENT[0], j + ORIENT[1]));
      T.at<uchar>(i, j) = abs(_p0 - _p1);
    }

=======
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  // padding 滤波计算
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++) {
        double _p0 = getGrayValue(I.at<Vec3b>(i, j));
        double _p1 = getGrayValue(I.at<Vec3b>(i + ORIENT[0], j + ORIENT[1]));
        T.at<uchar>(i, j) = abs(_p0 - _p1);
      }
  
>>>>>>> 9b9b5d5e6f2c45e444e8087a7ebeaca0c1a10e4b
  return T;
}

int main()
{
  Mat img = imread("imori.jpg", IMREAD_COLOR);
  int orient[] = {-1, 0};
  Mat A = differentialFilter(img, orient);
  imshow("before", img);
  imshow("differentialFilter", A);
  waitKey();
  return 0;
}