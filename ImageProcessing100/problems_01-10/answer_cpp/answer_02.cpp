/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-08 17:24:10
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_02.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat grayScale(Mat &I) {
  int nRows = I.rows;
  int nCols = I.cols;
  Mat T = Mat::zeros(nRows, nCols, CV_8U);
  for(int i = 0; i < nRows; i++)
    for(int j = 0; j < nCols; j++) {
      uchar b = I.at<Vec3b>(i, j)[0];
      uchar g = I.at<Vec3b>(i, j)[1];
      uchar r = I.at<Vec3b>(i, j)[2];
      T.at<uchar>(i, j) = (uchar)( 0.2126 * b + 0.7152 * g + 0.0722 * r );
    }
  return T;
}

int main() {
  Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
  Mat A = grayScale(img);
  imwrite("../imagelib/test_grayScale.jpg", A);
  imshow("grayScale", A);
  waitKey();
  return 0;
}