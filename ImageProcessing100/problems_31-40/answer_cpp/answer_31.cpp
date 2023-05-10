/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-06 10:13:15
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 16:30:40
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_31.cpp
 * @Description: 
 * 
 */
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
  Mat xy1 = (Mat_<double>(3, 1) << (double)_j, (double)_i, 1.0);
  Mat _xy1 = O * xy1;
  int _x = _xy1.at<double>(0, 0);
  int _y = _xy1.at<double>(1, 0);

  if(_x < 0 || _x >= I.cols || _y < 0 || _y >= I.rows)
    return 0;
  else
    return I.at<Vec3b>(_y, _x)[_c];
}

Mat afineTrans(const Mat &I, const Mat &O) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_rows, n_cols, CV_8UC3);
  Mat trans_mat = O.inv();

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      for(int c = 0; c < n_channel; c++)
        T.at<Vec3b>(i, j)[c] = getPixelValue(I, i, j, c, O);

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  imshow("before", img);
  double c_x = cvFloor((double)img.cols / 2.0) + 1;
  double c_y = cvFloor((double)img.rows / 2.0) + 1;
  double t_x = c_x - (c_x + 30.0 / img.rows * c_y);
  double t_y = c_y - c_y;
  Mat O = (Mat_<double>(3,3) << 1, 30.0 / img.rows, t_x, 0, 1, t_y, 0, 0, 1);
  // double c_x = cvFloor((double)img.cols / 2.0) + 1;
  // double c_y = cvFloor((double)img.rows / 2.0) + 1;
  // double t_x = c_x - c_x;
  // double t_y = c_y - (30.0 / img.cols * c_x + c_y);
  // Mat O = (Mat_<double>(3,3) << 1, 0, t_x, 30.0 / img.cols, 1, t_y, 0, 0, 1);
  Mat A = afineTrans(img, O);
  imshow("rotate and AfineTrans", A);
  waitKey();  return 0;
}