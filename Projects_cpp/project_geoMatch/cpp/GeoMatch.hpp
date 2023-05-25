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
  void findGeoMatchModel(float minScore = 0.7f, float greediness = 0.8f);
  void showModelDefined();
  void showMatchResult(float lowestScore = 0.5f);
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

  struct matchPoint {
    cv::Point coordiante;
    float score;
    cv::Mat affineMatrix;
    explicit matchPoint(cv::Point xy, float sc, cv::Mat afm = cv::Mat::zeros(3, 3, CV_32F))
        : coordiante(xy), score(sc), affineMatrix(afm) {}
  };

  cv::Mat sourceImage, tempImage;
  std::vector<GeoMatch::coorGradient> gradVecList;
  std::vector<GeoMatch::matchPoint> matchPointList;
  bool modelDefined, matchCompleted;
};

#endif  // OPENCV_GEOMATCH_HPP