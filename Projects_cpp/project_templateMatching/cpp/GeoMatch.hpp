#ifndef OPENCV_GEOMATCH_HPP
#define OPENCV_GEOMATCH_HPP

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

class GeoMatch {
 public:
  GeoMatch();
  void setSourceImage(const cv::Mat &sImage);
  void setTempImage(const cv::Mat &tImage);
  void processImage();
  void createGeoMatchModel(double minContrast, double maxContrast);
  void createGeoMatchModel(const cv::Mat &tempImage, double minContrast,
                           double maxContrast);
  cv::Mat getScoreMap();
  float findGeoMatchModel(float minScore, float greediness);
  void show();

 private:
  /// @brief containing coordiante, fx, fy, magnitude of Gradient Vector
  /// calculated by Canny methond
  struct coorGradient {
    cv::Point coordiante;
    cv::Vec2f edgesXY;
    explicit coorGradient(cv::Point xy, cv::Vec2f XY)
        : coordiante(xy), edgesXY(XY) {}
  };

  cv::Mat sourceImage, tempImage, matchMat;
  std::vector<GeoMatch::coorGradient> gradVecList;
  bool modelDefined;
};

#endif  // OPENCV_GEOMATCH_HPP