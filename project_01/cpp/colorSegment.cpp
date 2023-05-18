#include "colorSegment.hpp"
using namespace std;
using namespace cv;

void ColorSegment::setImage(const cv::Mat &image) {
  CV_Assert(image.type() == CV_8UC3);
  pImage = image.clone();
}

void ColorSegment::processImage() {
  cvtColor(pImage, hsvImage, COLOR_BGR2HSV);
}

void ColorSegment::colorCheck() {
  namedWindow("hsvImage", WINDOW_NORMAL);
  imshow("hsvImage", pImage);
  setMouseCallback("hsvImage", onMouse_colorSegment, this);
  waitKey();
  destroyWindow("hsvImage");
}

void ColorSegment::addColorRange(cv::String colorname, cv::Scalar lower, cv::Scalar upper) {
  colorList.push_back(colorRange(colorname, lower, upper));
}

void ColorSegment::createColorSegment() {
  for(const auto &i : colorList) {
    Mat newMask;
    inRange(hsvImage, i.lower, i.upper, i.mask);
    // 寻找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(i.mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    // 轮廓优化
    vector<vector<Point>> approxContours(contours.size());
    for(size_t i = 0; i < approxContours.size(); i++) {
      double epsilon = 0.0001 * arcLength(contours[i], 1);
      approxPolyDP(contours[i], approxContours[i], epsilon, 1);
    }
    // 遍历轮廓并绘制边界
    for (size_t i = 0; i < approxContours.size(); i++) {
      // 绘制外部边界
      drawContours(newMask, approxContours, i, Scalar(255), FILLED);
    }
    imshow("mask", newMask);
  }
  waitKey();
}

void ColorSegment::clearColorList() {
  colorList.clear();
}

void ColorSegment::onMouse_colorSegment(int event, int x, int y, int flags, void *userdata) {
  ColorSegment* self = static_cast<ColorSegment *>(userdata);
  if(event == EVENT_LBUTTONDOWN) {
    Vec3b pixel = self->hsvImage.at<Vec3b>(y, x);
    int h = pixel[0];
    int s = pixel[1];
    int v = pixel[2];
    cout << "Clicked at position (" << x << ", " << y << ")" << endl;
    cout << "HSV values: H=" << h << ", S=" << s << ", V=" << v << endl;
  }
} 