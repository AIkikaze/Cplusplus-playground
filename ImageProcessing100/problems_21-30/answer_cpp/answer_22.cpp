/*
author: wenzy
modified date: 20230505
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
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

  double getMeans() {
    double sum_val = 0.0;
    int num_val = 0;
    for(int i = 0; i < 256; i++)
      sum_val += val[i] * i, num_val += val[i];
    return sum_val / num_val;
  }

  double getstVar() {
    double v_means = getMeans();
    double st_var = 0.0;
    int num_val = 0;
    for(int i = 0; i < 256; i++)
      st_var += val[i] * (i - v_means) * (i - v_means), num_val += val[i];
    return sqrt(st_var / num_val);
  }

  void showHist(int BWID = 20, int HHEI = 500) {
    Mat A = Mat::zeros(HHEI, BWID * 256, CV_8UC3);
    double max_val = 0.0;
    for(int i = 0; i < 256; i++)
      max_val = fmax(max_val, val[i]);
    max_val *= 1.2;
    for(int i = 0; i < 256; i++)
      if(val[i]) {
        double _h = HHEI - (double)val[i] / max_val * HHEI;
        rectangle(A, Point(BWID*i, HHEI-1), Point(BWID*(i+1)-1, (int)_h), Scalar(255, 255, 255), FILLED);
      }
    imshow("hist", A);
    waitKey();
  }

  ~grayHist() {
    if(val != NULL)
      delete[] val;
  }
};

inline
double getGrayValue(Vec3b &pixel) {
  return 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];
}

grayHist makeHist(Mat &I) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  grayHist hist;

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      hist.val[(uchar)getGrayValue(I.at<Vec3b>(i, j))]++;

  return hist;
}

Mat histOperation(Mat &I, double M0 = 128.0, double S0 = 52.0) {
  CV_Assert(I.type() == CV_8UC3);

  int n_rows = I.rows;
  int n_cols = I.cols;
  int n_channel = I.channels();
  Mat T = Mat::zeros(n_rows, n_cols, CV_8UC3);
  grayHist H = makeHist(I);
  double h_means = H.getMeans();
  double h_stvar = H.getstVar();

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++)
      for(int c = 0; c < n_channel; c++) {
        double _pijc = I.at<Vec3b>(i, j)[c];
        _pijc = S0 / h_stvar * (_pijc - h_means) + M0;
        _pijc = fmin(_pijc, 255.0);
        _pijc = fmax(_pijc, 0.0);
        T.at<Vec3b>(i, j)[c] = (uchar)_pijc;
      }

  return T;
}

int main() {
  Mat img = imread("../imagelib/imori_dark.jpg", IMREAD_COLOR);
  imshow("before", img);
  // grayHist Hist = makeHist(img);
  // Hist.showHist(2, 360);
  Mat A = histOperation(img);
  imshow("histOperation", A);
  grayHist H = makeHist(A);
  H.showHist(2, 360);
  return 0;
}