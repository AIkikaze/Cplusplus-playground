/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-15 15:17:56
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-16 11:36:50
 * @FilePath: /C++-playground/project_01/cpp/model.cpp
 * @Description:
 *
 */
#include <opencv2/opencv.hpp>
using namespace cv;

void onMouse(int event, int x, int y, int flags, void *userdata)
{
  if (event == EVENT_LBUTTONDOWN)
  {
    Mat *image = static_cast<Mat *>(userdata);

    Vec3b pixel = image->at<Vec3b>(y, x);
    int h = pixel[0];
    int s = pixel[1];
    int v = pixel[2];

    std::cout << "Clicked at position (" << x << ", " << y << ")" << std::endl;
    std::cout << "HSV values: H=" << h << ", S=" << s << ", V=" << v << std::endl;
  }
}

void model()
{
  cv::Mat image = cv::imread("../imagelib/model_1.jpg", cv::IMREAD_COLOR);

  namedWindow("Image");
  imshow("Image", image);

  cv::Mat hsvImage;
  cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
  setMouseCallback("Image", onMouse, &hsvImage);

  // cv::Scalar lowerGreen = cv::Scalar(55, 90, 60);  // 绿色的下界
  // cv::Scalar upperGreen = cv::Scalar(65, 220, 140);  // 绿色的上界

  // cv::Mat mask;
  // cv::inRange(hsvImage, lowerGreen, upperGreen, mask);

  cv::Scalar lowerRed1 = cv::Scalar(0, 150, 150); // 红色的下界
  cv::Scalar upperRed1 = cv::Scalar(6, 220, 200); // 红色的上界
  std::cout << lowerRed1 << " " << upperRed1 << std::endl;

  cv::Mat mask1;
  cv::inRange(hsvImage, lowerRed1, upperRed1, mask1);

  cv::Scalar lowerRed2 = cv::Scalar(174, 150, 150); // 红色的下界
  cv::Scalar upperRed2 = cv::Scalar(180, 220, 200); // 红色的上界

  cv::Mat mask2;
  cv::inRange(hsvImage, lowerRed2, upperRed2, mask2);

  cv::Mat mask;
  bitwise_or(mask1, mask2, mask);

  int n = 5;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  for (int i = 0; i < n; i++)
  {
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  }

  // 寻找轮廓
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

  std::vector<std::vector<cv::Point>> approxContours(contours.size());
  for (size_t i = 0; i < contours.size(); i++)
  {
    double epsilon = 0.0001 * cv::arcLength(contours[i], true); // 设置逼近精度，这里使用轮廓周长的1%
    cv::approxPolyDP(contours[i], approxContours[i], epsilon, true);
  }

  // 遍历轮廓并绘制边界
  for (size_t i = 0; i < approxContours.size(); i++)
  {
    // 绘制外部边界
    cv::drawContours(image, approxContours, i, cv::Scalar(0, 255, 0), 2, cv::LINE_8, hierarchy, 0);
    // 绘制内部边界
    if (hierarchy[i][2] != -1)
    {
      cv::drawContours(image, approxContours, hierarchy[i][2], cv::Scalar(0, 255, 0), 2, cv::LINE_8, hierarchy, 0);
    }
  }

  cv::imshow("Contours", image);
  // cv::imwrite("../imagelib/green_1.jpg", image);
  cv::imwrite("../imagelib/red_1.jpg", image);
  waitKey(0);

  return;
}
