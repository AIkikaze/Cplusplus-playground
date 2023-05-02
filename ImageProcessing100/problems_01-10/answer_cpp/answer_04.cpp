/*
author: wenzy
modified date: 20230430
target: Binarize the image using Ostu's methond
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;
struct grayHist {
  int BASE;
  int* val;
  grayHist() {
    BASE = 256;
    val = new int [BASE];
    memset(val, 0, sizeof(int) * BASE);
  }
  grayHist(int base) {
    BASE = base;
    val = new int [BASE];
    memset(val, 0, sizeof(int) * BASE);
  }
  ~grayHist() {
    if(val != NULL)
      delete[] val;
  }
};

inline
uchar getGrayValue(Vec3b &pixel) {
  return (uchar)( 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0] );
}

grayHist makeHist(Mat &I) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  grayHist hist;

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      hist.val[getGrayValue(I.at<Vec3b>(i, j))]++;

  return hist;
}

uchar OstuMethod(const grayHist &H) {
  assert(H.BASE == 256);

  int num_tol = 0;
  long sum_tol = 0;
  int num_0 = 0;
  long sum_0 = 0;
  float mean_0 = 0.0f;
  float mean_1 = 0.0f;
  float w_0 = 0.0f;
  float s_inter = 0.0f;
  float s_inter_max = 0.0f;
  uchar bin_threshold_l = 0;
  uchar bin_threshold_r = 0;

  // 计算 numTol, sumTol
  for(int i = 0; i < H.BASE; i++) 
    num_tol += H.val[i], sum_tol += i*H.val[i];

  for(int t = 1; t < H.BASE; t++) {
    num_0 = 0;
    sum_0 = 0;
    for(int i = 0; i < t; i++)
      num_0 += H.val[i], sum_0 += i*H.val[i];
    mean_0 = (float)sum_0 / num_0;
    mean_1 = (float)(sum_tol - sum_0) / (num_tol - num_0);
    w_0 = (float)num_0 / num_tol;
    s_inter = w_0 * (1 - w_0) * (mean_0 - mean_1) * (mean_0 - mean_1);
    if(s_inter > s_inter_max) {
      s_inter_max = s_inter;
      bin_threshold_l = t;
    }
    else if(s_inter == s_inter_max) 
      bin_threshold_r = t;
  }

  return bin_threshold_l;
}

Mat Binarize(Mat &I, uchar bin_t) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  Mat T = Mat::zeros(n_rows, n_cols, CV_8U);

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++) {
      if(getGrayValue(I.at<Vec3b>(i, j)) < bin_t)
        T.at<uchar>(i, j) = 0;
      else
        T.at<uchar>(i, j) = 255;
    }
  
  return T;
}

int main() {
  Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
  grayHist Hist = makeHist(img);
  uchar binthresHold = OstuMethod(Hist);
  Mat A = Binarize(img, binthresHold);
  imshow("Binarize img with OstuMethod", A);
  waitKey();
  return 0;
}