/*
 * @Author: Alkikaze
 * @Date: 2023-05-02 17:46:28
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-07 13:08:30
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_10.cpp
 * @Description: 
 * 中值滤波
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

Mat medianFilter(Mat &I, Size size) {
  static int sizeTol = size.width * size.height;
  unique_ptr<uchar[]> arr(new uchar [sizeTol]);
  unique_ptr<int[]> dx(new int [sizeTol]);
  unique_ptr<int[]> dy(new int [sizeTol]);
  Mat T = Mat::zeros(I.size(), CV_8UC3);

  // 初始化移增量
  for(int i = 0; i < size.height; i++) {
    for(int j = 0; j < size.width; j++) {
      int _idx = i * size.width + j;
      dy[_idx] = -(size.height >> 1) + i;
      dx[_idx] = -(size.width >> 1) + j;
    }
  }

  // padding 滤波计算
  for(int i = 0; i < I.rows; i++)
    for(int j = 0; j < I.cols; j++)
      for(int c = 0; c < I.channels(); c++) {
        for(int idx = 0; idx < sizeTol; idx++) {
          if(i + dy[idx] < 0 || i + dy[idx] >= I.rows || j + dx[idx] < 0 || j + dx[idx] >= I.cols) {
            arr[idx] = 0;
            continue;
          }
          arr[idx] = I.at<Vec3b>(i + dy[idx], j + dx[idx])[c];
        }
        sort(arr.get(), arr.get() + sizeTol);
        T.at<Vec3b>(i, j)[c] = arr[(sizeTol >> 1) + 1];
      }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori_noise.jpg", IMREAD_COLOR);
  Mat A = medianFilter(img, Size(3, 3));
  imshow("before", img);
  imshow("medianFilter", A);
  imwrite("../imagelib/answer_10.jpg", A);
  waitKey();
  return 0;
}