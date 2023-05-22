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
cv::Rect box;
cv::Point startp, endp;
std::vector<cv::Point2d> stplist;
std::vector<cv::Point> holelist;
cv::Mat openedImage;

std::vector<cv::Vec6d> DetectEllipse(const cv::Mat &grey, const float &scale) {
  cv::Mat temp;
  // 图像缩放
  if (scale < 1.0)
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

  if (scale < 1.0) {
    // 还原椭圆位置
    double s = 1.0 / scale;
    for (auto &vec : ellipses) {
      // 还原椭圆尺寸
      for (int i = 0; i < 5; i++)
        vec[i] *= s;
    }
  }
  return ellipses;
}

void areCirclesApproximatelyConcentric(const std::vector<cv::Point2d> &circles, float deviationThreshold) {
  if (circles.empty()) {
    std::cout << "错误：未找到圆！" << std::endl;
    return;
  }

  // 计算圆心的平均值
  cv::Point2d averageCenter(0, 0);
  for (const auto &circle : circles) {
    averageCenter += circle;
  }
  averageCenter /= (double)(circles.size());

  // 计算圆心的偏差
  double deviation = 0.0;
  for (const auto &circle : circles) {
    deviation += cv::norm(circle - averageCenter);
  }
  deviation /= (double)(circles.size());
  
  // 输出计算结果
  std::cout << "平均圆心：" << startp + (cv::Point)averageCenter << std::endl;
  std::cout << "圆心标准差：" << deviation << std::endl;

  if (deviation > deviationThreshold)
    std::cout << "错误：找到的圆没有近似同心！" << std::endl;
  else {
    std::cout << "所有圆心偏差小于阈值，近似同心，求得圆心为：" << startp + (cv::Point)averageCenter << std::endl;
    holelist.push_back(startp + (cv::Point)averageCenter);
  }

  return; 
}

void findCircle(const cv::Mat &I) {
  stplist.clear();
  cv::Mat roiImage = I.clone();

  // cv::GaussianBlur(roiImage, roiImage, cv::Size(13, 13), 2.3);
  // cv::Canny(roiImage, roiImage, 30, 100);

  std::vector<cv::Vec6d> ellipses = DetectEllipse(roiImage, 1.0);

  cv::Mat outputImage;
  cv::cvtColor(roiImage, outputImage, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < ellipses.size(); i++)
  {
    cv::Point center((int)ellipses[i][0], (int)ellipses[i][1]);
    cv::Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
    double angle(ellipses[i][5]);
    cv::Scalar color = ellipses[i][2] == 0 ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0);
    // 过滤半径小于 30 的圆
    if (ellipses[i][2] && ellipses[i][2] < 30)
    {
      continue;
    }
    // 过滤某一轴长度小于 30 的椭圆
    else if (!ellipses[i][2] && (ellipses[i][3] < 30 || ellipses[i][4] < 30))
    {
      continue;
    }
    stplist.push_back(cv::Point2d(ellipses[i][0], ellipses[i][1]));
    // 在图像上绘制圆心
    cv::circle(outputImage, center, 5, cv::Scalar(0, 0, 255), -1);
    // 在图像上绘制圆，或椭圆形
    cv::ellipse(outputImage, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
  }

  areCirclesApproximatelyConcentric(stplist, 2.0);

  // 判断窗口是否已经创建
  int isVisible = cv::getWindowProperty("outputImage", cv::WND_PROP_VISIBLE);
  if (isVisible < 0) {
    // 窗口尚未创建，调用 namedWindow 创建窗口对象
    cv::namedWindow("outputImage", cv::WINDOW_NORMAL);
  }
  cv::imshow("outputImage", outputImage);
}

void __onMouse(int event, int x, int y, int flags, void *userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    selecting = true;
    startp = cv::Point(x, y);
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    box = cv::Rect(startp.x, startp.y, 0, 0);
  }
  else if (selecting && event == cv::EVENT_LBUTTONUP)
  {
    endp = cv::Point(x, y);
    selecting = false;

    cv::Rect roi(startp, endp);
    roiImage = (*((cv::Mat *)userdata))(roi);
    // 判断窗口是否已经创建
    int isVisible = cv::getWindowProperty("roi", cv::WND_PROP_VISIBLE);
    if (isVisible < 0) {
      // 窗口尚未创建，调用 namedWindow 创建窗口对象
      cv::namedWindow("roi", cv::WINDOW_NORMAL);
    }
    // 显示图片
    cv::imshow("roi", roiImage);
    findCircle(roiImage);

    if(holelist.size()) {
      std::cout << "定位孔位置：" << std::endl;
      for(auto &i : holelist) {
        std::cout << i << std::endl;
        // 在图像上绘制圆心
        cv::circle(openedImage, i, 5, cv::Scalar(0, 0, 255), -1);
      }
    }
    else
      std::cout << "错误：定位孔不准确！" << std::endl;

    cv::waitKey();
    cv::destroyWindow("roi");
    cv::destroyWindow("outputImage");
    cv::imshow("openedImage", openedImage);
  }
  else if(selecting && event == cv::EVENT_MOUSEMOVE) {
    box.width = x - startp.x;
    box.height = y - startp.y;
    cv::Mat tempImage;
    cv::cvtColor((*((cv::Mat *)userdata)), tempImage, cv::COLOR_GRAY2BGR);
    cv::rectangle(tempImage, box, cv::Scalar(0, 0, 255), 13); // 在临时图像上绘制矩形框
    cv::imshow("openedImage", tempImage);
  }
}

void circle()
{
  cv::Mat image = cv::imread("../imagelib/model_1.jpg", cv::IMREAD_GRAYSCALE);
  if (image.empty())
  {
    std::cout << "Failed to read image!" << std::endl;
    return;
  }

  // 应用盒式滤波
  cv::Mat filteredImage;
  cv::blur(image, filteredImage, cv::Size(13, 13)); // 使用13x13的盒式滤波器

  // 应用Otsu二值化
  cv::Mat binaryImage;
  cv::adaptiveThreshold(filteredImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 53, 2);

  // 形态学处理
  openedImage = binaryImage.clone();
  int n = 9;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  for (int i = 0; i < n; i++)
  {
    // cv::morphologyEx(openedImage, openedImage, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(openedImage, openedImage, cv::MORPH_CLOSE, kernel);
  }

  // 显示图像
  cv::namedWindow("openedImage", cv::WINDOW_NORMAL);
  cv::imshow("openedImage", openedImage);

  // 注册鼠标响应函数
  cv::setMouseCallback("openedImage", __onMouse, &openedImage);

  std::cout << "等待任意键...关闭图像窗口" << std::endl;
  cv::waitKey();

  return;
}
