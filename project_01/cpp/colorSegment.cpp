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
  windowName = "hsvImage";
  namedWindow(windowName, WINDOW_NORMAL);
  imshow(windowName, pImage);
  setMouseCallback(windowName, onMouse_colorSegment, this);
  waitKey();
  destroyAllWindows();
}

void ColorSegment::addColorRange(cv::String colorname, cv::Scalar lower, cv::Scalar upper) {
  colorList.push_back(colorRange(colorname, lower, upper));
}

void ColorSegment::__inRange(const cv::Mat &src, cv::Scalar lower, cv::Scalar upper, cv::Mat &dst) {
  if(lower[0] < upper[0]) 
    inRange(src, lower, upper, dst);
  else {
    Mat mask1;
    inRange(src, Scalar(0, lower[1], lower[2]), Scalar(upper[0], upper[1], upper[2]), mask1);
    Mat mask2;
    inRange(src, Scalar(lower[0], lower[1], lower[2]), Scalar(180, upper[1], upper[2]), mask2);
    bitwise_or(mask1, mask2, dst);
  }
}

void ColorSegment::createColorSegment() {
  for(auto &color : colorList) {
    Mat newMask = Mat::zeros(pImage.rows, pImage.cols, CV_8U);
    __inRange(hsvImage, color.lower, color.upper, color.mask);
    // 图像开运算处理：减少毛刺，光滑边缘
    int n = 7;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    for(int i = 0; i < n; i++) {
      morphologyEx(color.mask, color.mask, MORPH_OPEN, kernel);
    }
    // 寻找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(color.mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    // 轮廓优化
    vector<vector<Point>> approxContours(contours.size());
    for(size_t i = 0; i < approxContours.size(); i++) {
      double epsilon = 0.00005 * arcLength(contours[i], 1);
      approxPolyDP(contours[i], approxContours[i], epsilon, 1);
    }
    // 填充轮廓并输出在空白图像上
    drawContours(newMask, approxContours, -1, Scalar(255), FILLED);
    // 储存新蒙版
    color.mask = newMask.clone();
  }
}

void ColorSegment::copyTo(vector<String> &names, vector<Mat> &masks) {
  if(!masks.empty())
    masks.clear();
  if(!names.empty())
    names.clear();
  for(const auto &color : colorList) {
    masks.push_back(color.mask);
    names.push_back(color.colorname);
  }
}

void ColorSegment::showColorMasks() {
  for(const auto &color : colorList) {
    namedWindow(color.colorname, WINDOW_NORMAL);
    resizeWindow(color.colorname, 480, 640);
    imshow(color.colorname, color.mask);
  }
  waitKey();
  destroyAllWindows();
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
 