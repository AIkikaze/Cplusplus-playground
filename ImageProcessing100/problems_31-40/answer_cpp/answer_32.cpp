/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-06 13:44:27
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 15:48:21
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_32.cpp
 * @Description: 
 * 
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

Mat grayScale(Mat &I) {
  int n_row = I.rows;
  int n_col = I.cols;
  Mat T = Mat::zeros(n_row, n_col, CV_8U);

  for(int i = 0; i < n_row; i++) {
    for(int j = 0; j < n_col; j++) {
      double b = I.at<Vec3b>(i, j)[0];
      double g = I.at<Vec3b>(i, j)[1];
      double r = I.at<Vec3b>(i, j)[2];
      T.at<uchar>(i, j) = (uchar)( 0.2126 * b + 0.7152 * g + 0.0722 * r );
    }
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

void preinit(Mat &I, cMat &C) {
  // 将 img 转为灰度图
  I = grayScale(I);
  // 扩充图像以获得最佳计算尺寸
  int n_row = getOptimalDFTSize(I.rows);
  int n_col = getOptimalDFTSize(I.cols);
  copyMakeBorder(I, I, 0, n_row - I.rows, 0, n_col - I.cols, BORDER_CONSTANT, Scalar::all(0));
  // 初始化复数矩阵
  C = cMat(n_row, vector<complex<double>>(n_col, 0));
}

Mat mydft(const Mat &I) {
  // 将 planes 融合为多通道矩阵 Mat 同时用于储存傅里叶计算得到的复数矩阵
  Mat planes[] = { Mat_<double>(I), Mat::zeros(I.size(), CV_64F) };
  Mat mergeArray;
  merge(planes, 2, mergeArray);
  // 傅里叶变换
  dft(mergeArray, mergeArray);
  return mergeArray;
}

Mat fourierPlot(const Mat &I) {
  Mat T = I.clone();
  T += Scalar::all(1);
  log(T, T);
  // 将幅度谱剪裁为偶数行与偶数列(方便后面的重新排列）
	T = T(Range(0, T.rows & -2), Range(0, T.cols & -2));
	// 重新排列幅度谱的区域，使得幅度谱的原点位于图像中心
	int x0 = T.cols / 2;
	int y0 = T.rows / 2;
	Mat q0(T, Rect(0, 0, x0, y0));       //左上角图像
	Mat q1(T, Rect(x0, 0, x0, y0));      //右上角图像
	Mat q2(T, Rect(0, y0, x0, y0));      //左下角图像
	Mat q3(T, Rect(x0, y0, x0, y0));     //右下角图像
  // 交换第一象限和第三象限
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  // 交换第二象限和第四象限
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
  // 归一化
  normalize(T, T, 0, 1, NORM_MINMAX);
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  cMat fourier_coffix;

  preinit(img, fourier_coffix);
  Mat F = fourierTrans(img, fourier_coffix);
  Mat iF = invFourierTrans(fourier_coffix);
  Mat B = fourierPlot(F);

  imshow("grayScale", img);
  imshow("fourierTrans", B);
  imshow("invFourierTrans", iF);
  waitKey();

  return 0;
}