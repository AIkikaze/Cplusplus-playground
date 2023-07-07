/*
 * @Author: Alkikaze
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-06 15:55:19
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_01.cpp
 * @Description: 
 * 读取图像, 将 RGB 通道的图片转换为 BRG 通道
 */

#include <algorithm>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;

Mat channelSwap(Mat &I) {
  Mat T = I.clone();
  for (int r = 0; r < I.rows; r++)
    for (int c = 0; c < I.cols; c++)
      swap(T.at<Vec3b>(r, c)[0], T.at<Vec3b>(r, c)[2]);
  return T;
}

int main() {
  // 载入图片
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  // 调用函数
  Mat A = channelSwap(img);
  // 写入图片
  imwrite("../imagelib/answer_01.jpg", A);
  // 在窗口 "hello" 中显示图片
  imshow("answer_01", A);
  // 等待用户按下键盘
  waitKey();
  return 0;
}
