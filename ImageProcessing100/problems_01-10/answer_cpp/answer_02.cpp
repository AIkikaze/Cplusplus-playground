/*
 * @Author: Alkikaze
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-06 15:56:29
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_02.cpp
 * @Description: 
 * 将三通道的 BGR 图像转化为灰度图
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat grayScale(Mat &I) {
  Mat T = Mat::zeros(I.size(), CV_8U);
  for(int r = 0; r < I.rows; r++) {
    for(int c = 0; c < I.cols; c++) {
      uchar blue = I.at<Vec3b>(r, c)[0];
      uchar green = I.at<Vec3b>(r, c)[1];
      uchar red = I.at<Vec3b>(r, c)[2];
      T.at<uchar>(r, c) = (uchar)( 0.2126 * blue + 0.7152 * green + 0.0722 * red );
    }
  }
  return T;
}

int main() {
  // 读取图片
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  // 灰度值计算
  Mat A = grayScale(img);
  // 写入图片
  imwrite("../imagelib/answer_02.jpg", A);
  // 显示灰度图片
  imshow("answer_02", A);
  // 等待用户键入
  waitKey();
  return 0;
}