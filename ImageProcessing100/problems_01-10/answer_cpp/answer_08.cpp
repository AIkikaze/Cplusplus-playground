/*
 * @Author: Alkikaze
 * @Date: 2023-05-01 17:46:28
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-07 08:15:20
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_08.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
using namespace cv;
using namespace std;

inline
void maxPoolPixel(Mat &T, Mat &I, int row_begin, int col_begin, int kernel_size) {
  int max_inpool[] = {0, 0, 0};
  for(int i = row_begin; i < row_begin + 8; i++)
    for(int j = col_begin; j < col_begin + 8; j++)
      for(int c = 0; c < 3; c++)
        if(max_inpool[c] < I.at<Vec3b>(i, j)[c])
          max_inpool[c] = I.at<Vec3b>(i, j)[c];
  
  for(int i = row_begin; i < row_begin + 8; i++)
    for(int j = col_begin; j < col_begin + 8; j++)
      for(int c = 0; c < 3; c++)
        T.at<Vec3b>(i, j)[c] = max_inpool[c];
}

Mat maxPool(Mat &I, int kernel_size) {
  CV_Assert(I.type() == CV_8UC3);
  Mat T = Mat::zeros(I.size(), CV_8UC3);
  for(int i = 0; i < I.rows / kernel_size; i++) {
    for(int j = 0; j < I.cols / kernel_size; j++) {
      int idx_row = i * kernel_size;
      int idx_col = j * kernel_size;
      maxPoolPixel(T, I, idx_row, idx_col, kernel_size);
    }
  }
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = maxPool(img, 8);
  imshow("maxPool", A);
  imwrite("../imagelib/answer_08.jpg", A);
  waitKey();
  return 0;
}
