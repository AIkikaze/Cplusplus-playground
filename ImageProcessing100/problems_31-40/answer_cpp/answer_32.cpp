/*
author: wenzy
modified date: 20230506
*/
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

Mat histNormalize(Mat &I) {
  int n_rows = I.rows;
  int n_cols = I.cols;
  Mat T = Mat::zeros(n_rows, n_cols, CV_8U);
  double h_min = I.at<double>(0, 0);
  double h_max = 0.0;
  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++) {
      h_min = min(I.at<double>(i, j), h_min);
      h_max = max(I.at<double>(i, j), h_max);
    }

  for(int i = 0; i < n_rows; i++)
    for(int j = 0; j < n_cols; j++) {
        double _pij = I.at<double>(i, j);
        _pij = 255.0 * (_pij - h_min) / (h_max - h_min);
        T.at<uchar>(i, j) = (uchar)_pij;
      }

  return T;
}

Mat grayScale(Mat &I) {
  int n_row = I.rows;
  int n_col = I.cols;
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++) {
      double b = I.at<Vec3b>(i, j)[0];
      double g = I.at<Vec3b>(i, j)[1];
      double r = I.at<Vec3b>(i, j)[2];
      T.at<uchar>(i, j) = (uchar)( 0.2126 * b + 0.7152 * g + 0.0722 * r );
    }

  return T;
}

inline
double getFourierValue(const Mat &I, int k, int l) {
  double Iij;
  double theta;
  complex<double> val;
 
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Iij = (double)I.at<uchar>(i, j);
      theta = -2.0 * CV_PI * ((double)k*i/I.rows + (double)l*j/I.cols);
      val += complex<double>(cos(theta), sin(theta)) * Iij;
    }
  }
  val /= sqrt(I.rows * I.cols);

  return abs(val);
}

inline
double getInvFourierValue(const Mat &I, int k, int l) {
  double Iij;
  double theta;
  complex<double> val;
 
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Iij = (double)I.at<uchar>(i, j);
      theta = -2.0 * CV_PI * ((double)k*i/I.rows + (double)l*j/I.cols);
      val += complex<double>(cos(theta), sin(theta)) * Iij;
    }
  }
  val /= sqrt(I.rows * I.cols);

  return abs(val);
}

Mat fourierTrans(const Mat &I) {
  int n_row = I.rows;
  int n_col = I.cols;
  Mat G = Mat::zeros(n_row, n_col, CV_64F);
  
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++)
      G.at<double>(i, j) = getFourierValue(I, i, j);

  return G;
}

Mat invFourierTrans(const Mat &I) {
  int n_row = I.rows;
  int n_col = I.cols;
  Mat G = Mat::zeros(n_row, n_col, CV_8U);
  
  for(int i = 0; i < n_row; i++)
    for(int j = 0; j < n_col; j++)
      G.at<uchar>(i, j) = (uchar)getFourierValue(I, i, j);

  return G;
}

int main() {
  Mat img = imread("../imagelib/test.png", IMREAD_COLOR);

  Mat A = grayScale(img);
  Mat F = fourierTrans(A);
  Mat iF = invFourierTrans(F);
  Mat B = histNormalize(F);

  imshow("grayScale", A);
  imshow("fourierTrans", B);
  imshow("invFourierTrans", iF);
  waitKey();

  return 0;
}