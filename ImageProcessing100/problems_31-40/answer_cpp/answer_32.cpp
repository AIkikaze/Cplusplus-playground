/*
author: wenzy
modified date: 20230506
*/
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
using namespace std;
using namespace cv;
typedef vector<vector<complex<double>>> cMat;

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
complex<double> getFourierValue(const Mat &I, int k, int l) {
  double Iij;
  double theta;
  complex<double> val(0.0, 0.0);
 
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Iij = (double)I.at<uchar>(i, j);
      theta = -2.0 * CV_PI * ((double)k*i/I.rows + (double)l*j/I.cols);
      val += complex<double>(cos(theta), sin(theta)) * Iij;
    }
  }
  val /= sqrt(I.rows * I.cols);

  return val;
}

inline
double getInvFourierValue(const cMat &G, int y, int x) {
  int n_rows = G.size();
  int n_cols = n_rows > 0 ? G[0].size() : 0;
  double theta;
  complex<double> val(0.0, 0.0);
 
  for(int i = 0; i < n_rows; i++) {
    for(int j = 0; j < n_cols; j++) {
      theta = 2.0 * CV_PI * ((double)y*i/n_rows + (double)x*j/n_cols);
      val += complex<double>(cos(theta), sin(theta)) * G[i][j];
    }
  }
  val /= sqrt(n_rows * n_cols);

  return abs(val);
}

Mat fourierTrans(const Mat &I, cMat &C) {
  int n_row = I.rows;
  int n_col = I.cols;
  Mat G = Mat::zeros(n_row, n_col, CV_64F);
  
  for(int i = 0; i < n_row; i++) {
    for(int j = 0; j < n_col; j++) {
      C[i][j] = getFourierValue(I, i, j);
      G.at<double>(i, j) = abs(C[i][j]);
    }
  }

  return G;
}

Mat invFourierTrans(cMat &C) {
  int n_row = C.size();
  int n_col = n_row > 0? C[0].size() : 0;
  Mat I = Mat::zeros(n_row, n_col, CV_8U);

  for(int k = 0; k < n_row; k++) {
    for(int l = 0; l < n_col; l++) {
      I.at<uchar>(k, l) = uchar(getInvFourierValue(C, k, l));
    }
  }

  return I;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  int n_row = img.rows;
  int n_col = img.cols;
  cMat fourier_coffix(n_row, vector<complex<double>>(n_col, 0));

  Mat A = grayScale(img);
  Mat F = fourierTrans(A, fourier_coffix);
  Mat iF = invFourierTrans(fourier_coffix);
  Mat B = histNormalize(F);

  imshow("grayScale", A);
  imshow("fourierTrans", B);
  imshow("invFourierTrans", iF);
  waitKey();

  return 0;
}