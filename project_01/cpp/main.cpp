/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-15 15:17:56
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-16 13:20:47
 * @FilePath: /C++-playground/project_01/cpp/main.cpp
 * @Description:
 *
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;

int lasty = 0;
int lastx = 0;
void onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    cv::Mat *image = static_cast<cv::Mat *>(userdata);
    cout << "point:" << y << "," << x << endl;
    cout << "gray on point:" << (int)image->at<uchar>(y, x) << endl;
    double dist = sqrt(pow(lastx - x, 2) + pow(lasty - y, 2));
    cout << "dist from last to now:" << dist << endl;
    lastx = x;
    lasty = y;
  }
}

int main()
{
  cv::Mat image = cv::imread("../imagelib/test.TIFF", cv::IMREAD_GRAYSCALE);
  cv::imshow("Original Image", image);

  // 应用盒式滤波
  cv::Mat filteredImage;
  cv::blur(image, filteredImage, cv::Size(13, 13)); // 使用3x3的盒式滤波器

  // 应用Otsu二值化
  cv::Mat binaryImage;
  cv::threshold(filteredImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // 形态学处理
  int n = 5;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  for (int i = 0; i < n; i++)
  {
    cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);
  }
  cv::imshow("Opened Image", binaryImage);
  cv::setMouseCallback("Opened Image", onMouse, &binaryImage);
  cv::waitKey();

  // // 寻找轮廓
  // cv::Mat contourImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  // // cv::cvtColor(contourImage, contourImage, cv::COLOR_GRAY2BGR);
  // std::vector<std::vector<cv::Point>> contours;
  // std::vector<cv::Vec4i> hierarchy;
  // cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

  // std::vector<std::vector<cv::Point>> approxContours(contours.size());
  // for (size_t i = 0; i < contours.size(); i++)
  // {
  //   double epsilon = 0.0001 * cv::arcLength(contours[i], true); // 设置逼近精度，这里使用轮廓周长的1%
  //   cv::approxPolyDP(contours[i], approxContours[i], epsilon, true);
  // }

  // // 遍历轮廓并绘制边界
  // for (int i = 0; i < approxContours.size(); i++)
  // {
  //   // 绘制外部边界
  //   cv::drawContours(contourImage, approxContours, i, cv::Scalar(0, 0, 255), 2, cv::LINE_8, hierarchy, 0);
  //   // 绘制内部边界
  //   if (hierarchy[i][2] != -1)
  //   {
  //     cv::drawContours(contourImage, approxContours, hierarchy[i][2], cv::Scalar(0, 255, 0), 2, cv::LINE_8, hierarchy, 0);
  //   }
  // }


  // cv::imshow("Contour Image", contourImage);
  // cv::waitKey(0);

  return 0;
}
