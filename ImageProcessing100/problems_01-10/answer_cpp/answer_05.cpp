/*
auther: wenzy
modified date: 20230501
target: Implement HSV conversion and flip Hue H.
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void hueConverePixel(Vec3b &p) {
  int max_bgr = 0;
  int min_bgr = 256;
  float hsv_h = -1.0f;
  int hsv_s = 0;
  int hsv_v = 0;

  // 计算 min_bgr, max_bgr
  for(int i = 0; i < 3; i++) {
    if(p[i] > max_bgr) 
      max_bgr = p[i];
    if(p[i] < min_bgr)
      min_bgr = p[i];
  }

  // 计算色彩模型 HSV 下的各个分量
  hsv_v = max_bgr;
  hsv_s = max_bgr - min_bgr;
  if(min_bgr == max_bgr) 
    hsv_h = 0.0f;
  else if(min_bgr == p[0]) 
    hsv_h = 60.0f * (p[1] - p[2]) / (max_bgr - min_bgr) + 60.0f;
  else if(min_bgr == p[1]) 
    hsv_h = 60.0f * (p[2] - p[0]) / (max_bgr - min_bgr) + 300.0f;
  else     
    hsv_h = 60.0f * (p[0] - p[1]) / (max_bgr - min_bgr) + 180.0f;
  
  // 交换色相
  hsv_h = fmod(180.0f + hsv_h, 360.0f);

  // HSV2BGR
  float delta[3] = {0, 0, 0};
  int _c = hsv_s;
  float _h = hsv_h / 60.0f;
  float _x = _c * (1 - abs(fmod(_h, 2.0f)- 1));
  if(hsv_h == -1.0f) ; // do nothing
  else if(_h >= 0 && _h < 2) {
    delta[0] = _c;
    delta[1] = _x; 
    if(_h >= 1) 
      swap(delta[0], delta[1]);
  }
  else if(_h >= 2 && _h < 4) {
    delta[1] = _c;
    delta[2] = _x;
    if(_h >= 3)
      swap(delta[1], delta[2]);
  }
  else {
    delta[2] = _c;
    delta[0] = _x;
    if(_h >= 5)
      swap(delta[2], delta[0]);
  }
  for(int i = 0; i < 3; i++)
    p[i] = (uchar)((hsv_v - _c) + delta[i]);
  swap(p[0], p[2]);
}

Mat hueFilp(Mat &I) {
  CV_Assert(I.type() == CV_8UC3);

  int n_row = I.rows;
  int n_col = I.cols;
  Mat T = I.clone();

  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++)
      hueConverePixel(T.at<Vec3b>(i, j));

  return T;
}

int main() {
  Mat img = imread("../imagelib/test_0.jpg", IMREAD_COLOR);
  Mat A = hueFilp(img);
  imshow("phaseSwap", A);
  waitKey();
  return 0;
}