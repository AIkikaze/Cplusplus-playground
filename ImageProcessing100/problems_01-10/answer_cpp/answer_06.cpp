/*
auther: wenzy
modified date: 20230501
target: Discretization of Color
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <bitset>
using namespace std;
using namespace cv;

inline 
void colorDisPixel(Vec3b &p) {
  uchar color_bush[] = {32, 96, 160, 224};
  int color_threshold[] = {0, 63, 127, 191, 256};
  for(int i = 0; i < 3; i++) 
    for(int k = 0; k < 4; k++)
      if(p[i] >= color_threshold[k] && p[i] < color_threshold[k+1])
        p[i] = color_bush[k];
}

Mat colorDiscretize(Mat &I) {
  CV_Assert(I.type() == CV_8UC3);

  int n_row = I.rows;
  int n_col = I.cols;
  Mat T = I.clone();

  for(int i = 0; i < n_row; i++) 
    for(int j = 0; j < n_col; j++)
      colorDisPixel(T.at<Vec3b>(i, j));
    
  return T;
}

int main() {
  Mat img = imread("../imagelib/test_0.jpg", IMREAD_COLOR);
  Mat A = colorDiscretize(img);
  imshow("colorDiscretize", A);
  waitKey();
  return 0;
}