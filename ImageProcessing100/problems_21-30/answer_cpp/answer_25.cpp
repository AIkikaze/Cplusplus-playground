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

Mat nestNeiInterpolate(Mat &I, double ZOOM = 1.5) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows * ZOOM;
  int n_cols = I.cols * ZOOM;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_rows, n_cols, CV_8UC3);

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      for(int c = 0; c < n_channel; c++) {
        int _i = (double)i/ZOOM + 0.5;
        int _j = (double)j/ZOOM + 0.5;
        T.at<Vec3b>(i, j)[c] = I.at<Vec3b>(_i, _j)[c];
      }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  imshow("before", img);
  Mat A = nestNeiInterpolate(img);
  imshow("nestNeiInterpolate", A);
  return 0;
}