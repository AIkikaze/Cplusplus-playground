/*
 * @Author: Alkikaze
 * @Date: 2023-05-01 17:46:28
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-06 17:23:23
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_06.cpp
 * @Description: 
 * 对三通道 8 bit 图像进行色彩量化，得到三通道 2 bit 图像。
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <bitset>
using namespace std;
using namespace cv;

inline 
void colorDisPixel(Vec3b &p) {
  uchar color_bush[] = {32, 96, 160, 224};
  int color_threshold[] = {0, 63, 127, 191, 256};
  for(int i = 0; i < 3; i++) 
    for(int k = 0; k < 4; k++)
      if(p[i] >= color_threshold[k] && p[i] < color_threshold[k+1])
        p[i] = color_bush[k];
}

Mat colorDiscretize(Mat &I) {
  CV_Assert(I.type() == CV_8UC3);
  Mat T = I.clone();
  for(int i = 0; i < I.rows; i++) 
    for(int j = 0; j < I.cols; j++)
      colorDisPixel(T.at<Vec3b>(i, j));
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat A = colorDiscretize(img);
  imshow("answer_06", A);
  imwrite("../imagelib/answer_06.jpg", A);
  waitKey();
  return 0;
}