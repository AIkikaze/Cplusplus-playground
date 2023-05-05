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

double biInterPixel(const Mat &I, int X, int Y, int C, double R, int B_TYPE) {
  int dij[] = { 0, 1 };
  double i = (double)(X+0.5) / R - 0.5;
  double j = (double)(Y+0.5) / R - 0.5;
  double dx = i - cvFloor(i);
  double dy = j - cvFloor(j);
  double pixel[4];

  for(int k = 0; k < 4; k++)
    pixel[k] = I.at<Vec3b>(borderInterpolate((int)i + dij[k/2], I.rows, B_TYPE), 
    borderInterpolate((int)j + dij[k%2], I.cols, B_TYPE))[C];
  
  return (1 - dx) * (1 - dy) * pixel[0] + dy * (1 - dx) * pixel[1] + dx * (1 - dy) * pixel[2] + dx * dy * pixel[3];
}

Mat bilinearInterpol(const Mat &I, double ZOOM = 1.5) {
  CV_Assert(I.type() == CV_8UC3);

  int n_Rows = I.rows * ZOOM;
  int n_Cols = I.cols * ZOOM;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_Rows, n_Cols, CV_8UC3);

  for(int i = 0; i < n_Rows; i++)
    for(int j = 0; j < n_Cols; j++)
      for(int c = 0; c < n_channel; c++)
        T.at<Vec3b>(i, j)[c] = biInterPixel(I, i, j, c, ZOOM, BORDER_REFLECT);

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  imshow("before", img);
  Mat A = bilinearInterpol(img);
  imshow("bilinearInterpol", A);
  return 0;
}