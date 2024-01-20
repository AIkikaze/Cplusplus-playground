#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

class Detector {
public:
  int TargeType;
  int CoefficientType;

  void process();
private:
  Mat sobel_x;
  Mat sobel_y;
  Mat M;
};

int main() {
  Mat image = imread("../imagelib/lenna.jpg", IMREAD_COLOR);

  Mat image_gray;
  cvtColor(image, image_gray, COLOR_BGR2GRAY);

  Mat harris;
  cornerHarris(image_gray, harris, 2, 3, 0.04);

  Mat normalized;
  normalize(harris, normalized, 0, 255, NORM_MINMAX, CV_8U);

  // 寻找 harris 角点
  vector<KeyPoint> points;
  for (int r = 0; r < normalized.rows; r++) {
    for (int c = 0; c < normalized.cols; c++) {
      uchar R = normalized.at<uchar>(r, c);
      if (R > 125) {
        points.emplace_back(c, r, 3);
      }
    }
  }

  Mat output;
  drawKeypoints(image, points, output);
  imshow("harris", output);
  waitKey();
  return 0;
}