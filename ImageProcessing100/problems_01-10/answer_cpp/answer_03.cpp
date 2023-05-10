/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-08 17:27:47
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_03.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

inline
uchar getGrayValue(Vec3b &pixel) {
  return (uchar)( 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0] );
}

Mat Binarize(Mat &I, uchar bin_t = 128) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  Mat T = Mat::zeros(n_rows, n_cols, CV_8U);

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++) {
      if(getGrayValue(I.at<Vec3b>(i, j)) < bin_t)
        T.at<uchar>(i, j) = 0;
      else
        T.at<uchar>(i, j) = 255;
    }
  
  return T;
}

int main() {
  Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
  Mat A = Binarize(img);
  imshow("Binarize", A);
  waitKey();
  return 0;
}