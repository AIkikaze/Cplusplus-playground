/*
 * @Author: Alkikaze
 * @Date: 2023-05-01 17:46:28
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-07 08:07:05
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_07.cpp
 * @Description: 
 * 以 8x8 大小的核进行平均池化
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
using namespace cv;
using namespace std;

inline
void averagePoolPixel(Mat &T, Mat &I, int row_begin, int col_begin, int kernel_size) {
  int sum_tol[] = {0, 0, 0};
  for(int i = row_begin; i < row_begin + 8; i++)
    for(int j = col_begin; j < col_begin + 8; j++)
      for(int c = 0; c < 3; c++)
        sum_tol[c] += I.at<Vec3b>(i, j)[c];
  
  for(int i = row_begin; i < row_begin + 8; i++)
    for(int j = col_begin; j < col_begin + 8; j++)
      for(int c = 0; c < 3; c++)
        T.at<Vec3b>(i, j)[c] = (float) sum_tol[c] / 64;
}

Mat averagePool(Mat &I, int kernel_size) {
  CV_Assert(I.type() == CV_8UC3);
  Mat T = Mat::zeros(I.size(), CV_8UC3);
  for(int i = 0; i < I.rows / kernel_size; i++) {
    for(int j = 0; j < I.cols / kernel_size; j++) {
      int idx_row = i * kernel_size;
      int idx_col = j * kernel_size;
      averagePoolPixel(T, I, idx_row, idx_col, kernel_size);
    }
  }
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = averagePool(img, 8);
  imshow("averagePool", A);
  imwrite("../imagelib/answer_07.jpg", A);
  waitKey();
  return 0;
}
