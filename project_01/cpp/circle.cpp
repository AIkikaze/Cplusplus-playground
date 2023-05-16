/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-16 12:07:44
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-16 14:06:18
 * @FilePath: \Cplusplus-playground\project_01\cpp\circle.cpp
 * @Description:
 *
 */
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <vector>
cv::Mat roiImage;
bool selecting = false;
cv::Point startp, endp;

std::vector<cv::Vec6d> DetectEllipse(const cv::Mat &grey, const float &scale) {
  cv::Mat temp;
  // 图像缩放
  if(scale < 1.0)
    resize(grey, temp, cv::Size(grey.cols * scale, grey.rows * scale));
  else
    temp = grey;
  
  // 创建 edge drawing 对象 ed 用于检测椭圆
  cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
  ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::PREWITT;
  ed->params.GradientThresholdValue = 20;
  ed->params.AnchorThresholdValue = 2;

  // 检测边缘
  ed->detectEdges(temp);
  std::vector<cv::Vec4f> lines;
  // 检测直线，检测直线前需边缘检测
  ed->detectLines(lines);
  // 检测圆与椭圆，检测圆前需要检测直线
  std::vector<cv::Vec6d> ellipses;
  ed->detectEllipses(ellipses);

  if(scale < 1.0) {
    // 还原椭圆位置
    double s = 1.0 / scale;
    for(auto &vec : ellipses) {
      // 还原椭圆尺寸
      for(int i = 0; i < 5; i++)
        vec[i] *= s;
    }
  }
  return ellipses;
}


void findCircle(const cv::Mat &I) {
  cv::Mat roiImage = I.clone();

  cv::GaussianBlur(roiImage, roiImage, cv::Size(7, 7), 1.5);
  cv::Canny(roiImage, roiImage, 50, 100);

  std::vector<cv::Vec6d> ellipses = DetectEllipse(roiImage, 1.0);

  cv::Mat outputImage;
  cv::cvtColor(roiImage, outputImage, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < ellipses.size(); i++) {
      cv::Point center((int)ellipses[i][0], (int)ellipses[i][1]);
      cv::Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
      double angle(ellipses[i][5]);
      cv::Scalar color = ellipses[i][2] == 0 ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0);
      cv::ellipse(outputImage, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
  }
  cv::namedWindow("outputImage", cv::WINDOW_NORMAL);
  cv::imshow("outputImage", outputImage);
}

void onMouse(int event, int x, int y , int flags, void* userdata) {
  if(event == cv::EVENT_LBUTTONDOWN) {
    selecting = true;
    startp = cv::Point(x, y);
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  }
  else if(selecting && event == cv::EVENT_LBUTTONUP) {
    endp = cv::Point(x, y);
    selecting = false;

    cv::Rect roi(startp, endp);
    roiImage = (*((cv::Mat*)userdata))(roi);
    cv::namedWindow("roi", cv::WINDOW_NORMAL);
    cv::imshow("roi", roiImage);
    findCircle(roiImage);
  }
}

int main() {
  cv::Mat image = cv::imread("../imagelib/test.TIFF", cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cout << "Failed to read image!" << std::endl;
    return -1;
  }

  // 应用盒式滤波
  cv::Mat filteredImage;
  cv::blur(image, filteredImage, cv::Size(7, 7)); // 使用3x3的盒式滤波器

  // 应用Otsu二值化
  cv::Mat binaryImage;
  cv::adaptiveThreshold(filteredImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 53, 2);

  // 形态学处理
  cv::Mat openedImage = binaryImage.clone();
  int n = 7;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
  for (int i = 0; i < n; i++) {
    // cv::morphologyEx(openedImage, openedImage, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(openedImage, openedImage, cv::MORPH_CLOSE, kernel);
  }

  // 显示图像
  cv::namedWindow("input", cv::WINDOW_NORMAL);
  cv::imshow("input", image);
  cv::namedWindow("openedImage", cv::WINDOW_NORMAL);
  cv::imshow("openedImage", openedImage);

  // 注册鼠标响应函数
  cv::setMouseCallback("openedImage", onMouse, &openedImage);

  cv::waitKey();

  return 0;
}
