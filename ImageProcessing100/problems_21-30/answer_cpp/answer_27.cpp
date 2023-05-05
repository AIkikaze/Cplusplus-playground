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
double getHt(double t, double a = -1.0) {
  double result = 0.0;
  if(t <= 1)
    result = (a+2) * pow(t, 3) - (a+3) * pow(t, 2) + 1;
  else if(t <= 2) 
    result = a * pow(t, 3) - 5 * a * pow(t, 2) + 8 * a * t - 4 * a;
  else ;
  return result;
}

inline 
double getBicubicInter(const Mat &I, int X, int Y, int C, double R, int B_TYPE) {
  int dij[] = { -1, 0, 1, 2 };
  double i = (double)(X+0.5) / R - 0.5;
  double j = (double)(Y+0.5) / R - 0.5; 
  double sum_h = 0.0;
  double dx[4], hx[4];
  double dy[4], hy[4];
  double p[16];
  double result = 0.0;

  for(int k = 0; k < 4; k++) {
    dx[k] = abs(i - (cvFloor(i) + dij[k]));
    dy[k] = abs(j - (cvFloor(j) + dij[k]));
    hx[k] = getHt(dx[k]);
    hy[k] = getHt(dy[k]);
  }

  for(int k = 0; k < 16; k++) {
    sum_h += hx[k/4] * hy[k%4];
    p[k] = I.at<Vec3b>(borderInterpolate((int)i + dij[k/4], I.rows, B_TYPE), borderInterpolate((int)j + dij[k%4], I.cols, B_TYPE))[C];
    result += p[k] * hx[k/4] * hy[k%4];
  }
  result /= sum_h;
  result = fmax(result, 0.0);
  result = fmin(result, 255.0);

  return result;
}

Mat bicubicInterpol(const Mat &I, double ZOOM = 1.5) {
  CV_Assert(I.type() == CV_8UC3);

  int n_Rows = I.rows * ZOOM;
  int n_Cols = I.cols * ZOOM;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_Rows, n_Cols, CV_8UC3);

  for(int i = 0; i < n_Rows; i++)
    for(int j = 0; j < n_Cols; j++)
      for(int c = 0; c < n_channel; c++)
        T.at<Vec3b>(i, j)[c] = getBicubicInter(I, i, j, c, ZOOM, BORDER_REFLECT);

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  imshow("before", img);
  Mat A = bicubicInterpol(img, 2.0);
  imshow("bicubicInterpol", A);
  return 0;
}