/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-08 15:34:54
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 14:45:43
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_39.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;

Mat rgb2YCbCr(const Mat &I) {
  Mat T = Mat::zeros(I.rows, I.cols, CV_8UC3);

  double Y, Cb, Cr;
  double R, G, B;
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Vec3b p = I.at<Vec3b>(i, j);
      Y = 0.299 * p[2] + 0.5870 * p[1] + 0.114 * p[0];
      Cb = - 0.1687 * p[2] - 0.3323 * p[1] + 0.5 * p[0] + 128;
      Cr = 0.5 * p[2] - 0.4187 * p[1] - 0.0813 * p[0] + 128;
      Y *= 0.7;
      R = Y + (Cr - 128) * 1.4102;
      G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139;
      B = Y + (Cb - 128) * 1.7718;
      T.at<Vec3b>(i, j)[0] = (uchar)B;
      T.at<Vec3b>(i, j)[1] = (uchar)G;
      T.at<Vec3b>(i, j)[2] = (uchar)R;    
    }
  }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);

  Mat A = rgb2YCbCr(img);

  imshow("origin", img);
  imshow("YCbCr", A);
  waitKey();
  return 0;
}