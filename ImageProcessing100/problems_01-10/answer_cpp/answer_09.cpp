/*
 * @Author: Alkikaze
 * @Date: 2023-05-04 20:33:54
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-07 09:11:59
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_09.cpp
 * @Description: 
 * 高斯滤波
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

Mat gaussianFilter(Mat &I, Size size, double sigma) {
  static int sizeTol = size.height * size.width;
  double kernel_sum = 0.0;
  unique_ptr<double[]> kernel(new double [sizeTol]);
  unique_ptr<int[]> dx(new int [sizeTol]);
  unique_ptr<int[]> dy(new int [sizeTol]);
  Mat T = Mat::zeros(I.size(), CV_8UC3);

  // 初始化高斯滤波器核与位移增量
  for(int i = 0; i < size.width; i++) {
    for(int j = 0; j < size.height; j++) {
      int idx = i * size.width + j;
      dy[idx] = -(size.height>>1) + i;
      dx[idx] = -(size.width>>1) + j;
      double xy2 = (double)dx[idx] * dx[idx] + (double)dy[idx] * dy[idx];
      kernel[idx] = exp( - xy2 / (2 * sigma * sigma) ) / ( 2 * CV_PI * sigma * sigma );
      kernel_sum += kernel[idx];
    }
  }
  
  // 归一化处理
  for(int i = 0; i < sizeTol; i++)
    kernel[i] /= kernel_sum;

  // padding 滤波计算
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      for(int c = 0; c < I.channels(); c++) {
        double pixel_result = 0.0;
        for(int idx = 0; idx < sizeTol; idx++) {
          if(i + dy[idx] < 0 || i + dy[idx] >= I.rows || j + dx[idx] < 0 || j + dx[idx] >= I.cols)
            continue;
          pixel_result += kernel[idx] * I.at<Vec3b>(i + dy[idx], j + dx[idx])[c];
        }
        T.at<Vec3b>(i, j)[c] = (uchar)pixel_result;
      }
    }
  }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori_noise.jpg", IMREAD_COLOR);
  Mat A = gaussianFilter(img, Size(3, 3), 1.3);
  imshow("before", img);
  imshow("gaussianFilter", A);
  imwrite("../imagelib/answer_09.jpg", A);
  waitKey();
  return 0;
}