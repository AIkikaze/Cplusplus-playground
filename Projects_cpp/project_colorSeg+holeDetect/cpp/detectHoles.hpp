#ifndef OPENCV_DETECTHOLES_HPP
#define OPENCV_DETECTHOLES_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include <iostream>

// 定义类：用于圆孔定位
class HoleDetector {
public:
  struct HoleDetector_Params {
    int size_blur;
    int size_adaptive_binary_block;
    double c_adaptive_const;
    int times_morphologyEx;
    int size_structeElement;
    int type_morphologyEx;
    int size_holeImage;
  }params;

  HoleDetector();
  // ~HoleDetector();

  void setImage(const cv::Mat& image);
  void processImage();
  void drawResult();
  void drawHoleList(const cv::String &winname, int flag);
  void addNewHole();
  void copyHolelistTo(std::vector<cv::Point> &dst);
  void copyHolelistImageTo(std::vector<cv::Mat> &dst);
  int getSize();

private:
  cv::Mat pImage, roiImage, holeImage;
  std::vector<cv::Point> holelist;
  std::vector<cv::Mat> holelist_image;
  bool selecting, afterProcess, holeFound;
  cv::Point start_point, end_point;
  cv::Rect box_selecting;

  void __setMouseCallback(const cv::String &winname);
  static void onMouse(int event, int x, int y, int flags, void *userdata);
  std::vector<cv::Vec6d> detectEllipse(const cv::Mat &grey, const double &scale);
  void findHoles();
  bool areCirclesApproximatelyConcentric(const std::vector<cv::Point2d> &circles, double deviationThreshold);
};


#endif // OPENCV_DETECTHOLES_HPP