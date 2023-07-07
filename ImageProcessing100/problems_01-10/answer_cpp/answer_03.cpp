/*
 * @Author: Alkikaze
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-06 16:04:13
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_03.cpp
 * @Description: 
 * 对图像进行二值化
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

Mat Binarize(Mat &I, uchar threshold = 128) {
  CV_Assert(I.type() == CV_8UC3);
  Mat T = Mat::zeros(I.size(), CV_8U);
  for(int r = 0; r < I.rows; r++) {
    for(int c = 0; c < I.cols; c++) {
      int flag = getGrayValue(I.at<Vec3b>(r, c)) > threshold;
      T.at<uchar>(r, c) = 255 * flag;
    }
  }
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = Binarize(img);
  imshow("answer_03", A);
  imwrite("../imagelib/answer_03.jpg", A);
  waitKey();
  return 0;
}