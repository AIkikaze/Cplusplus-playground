/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-10 08:54:37
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-10 09:05:29
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_41-50\answer_cpp\answer_41.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat gussianFilter(const Mat &I, Size Kernel_size, double Sigma) {
  Mat T = Mat::zeros(I.size(), I.type());

}

int main() {
  // 读取图像
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  // 对图像进行灰度化处理
  cvtColor(img, img, COLOR_BGR2GRAY);
  // 高斯滤波
  Mat A = guassianFilter(img, Size(5, 5), 1.4);
  imshow("img", img);
  waitKey();
  return 0;
}