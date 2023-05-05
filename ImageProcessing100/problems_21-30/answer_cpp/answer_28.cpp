/*
author: wenzy
modified date: 20230505
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;
using namespace cv;

inline 
uchar getPixelValue(const Mat &I, int _i, int _j, int _c, const Mat &O) {
  Mat xy1 = (Mat_<double>(3, 1) << (double)_i, (double)_j, 1.0);
  Mat _xy1 = O * xy1;
  int _x = _xy1.at<double>(0, 0);
  int _y = _xy1.at<double>(1, 0);

  if(_x < 0 || _x >= I.rows || _y < 0 || _y >= I.cols)
    return 0;
  else
    return I.at<Vec3b>(_x, _y)[_c];
}

Mat afineTrans(const Mat &I, const Mat &O) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_rows, n_cols, CV_8UC3);

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      for(int c = 0; c < n_channel; c++)
        T.at<Vec3b>(i, j)[c] = getPixelValue(I, i, j, c, O);

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  imshow("before", img);
  Mat O = (Mat_<double>(3,3) << 1, 0, 30, 0, 1, -30, 0, 0, 1);
  Mat A = afineTrans(img, O);
  imshow("afineTrans", A);
  waitKey();
  return 0;
}