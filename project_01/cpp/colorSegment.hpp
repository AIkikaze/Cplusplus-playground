#ifndef OPENCV_COLORSEGMENT_HPP
#define OPENCV_COLORSEGMENT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class ColorSegment {
public:

  void setImage(const cv::Mat &image);
  void processImage();
  void colorCheck();
  void addColorRange(cv::String colorname, cv::Scalar lower, cv::Scalar upper);
  void __inRange(const cv::Mat &src, cv::Scalar lower, cv::Scalar upper, cv::Mat &dst);
  void createColorSegment();
  void copyTo(std::vector<cv::String> &names, std::vector<cv::Mat> &masks);
  void showColorMasks();
  void clearColorList();

private:
  struct colorRange {
    cv::String colorname;
    cv::Scalar lower, upper;
    cv::Mat mask;
    colorRange(cv::String x, cv::Scalar l, cv::Scalar u):
      colorname(x), lower(l), upper(u) { }
  };

  cv::Mat pImage, hsvImage;
  std::vector<ColorSegment::colorRange> colorList;
  cv::String windowName;
  double zoomFactor; 
  
  static void onMouse_colorSegment(int event, int x, int y, int flags, void *userdata);
};

#endif // OPENCV_COLORSEGMENT_HPP