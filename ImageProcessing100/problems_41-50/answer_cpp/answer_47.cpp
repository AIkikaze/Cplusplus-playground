#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
using namespace std;
using namespace cv;

struct grayHist {
  vector<int> val;
  grayHist(int base) {
    val = vector<int>(base, 0);
  }
  void makeHist(const Mat &I) {
    CV_Assert(I.type() == CV_8U);
    for(int i = 0; i < I.rows; i++)
      for(int j = 0; j < I.cols; j++)
        val[(int)I.at<uchar>(i, j)]++;
  }
  uchar OstuMethod() {
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
    for(int i = 0; i < 256; i++) 
      num_tol += val[i], sum_tol += i*val[i];
    // 计算并记录 s_inter, s_inter_max 对应的灰度值
    for(int t = 1; t < 256; t++) {
      num_0 = 0;
      sum_0 = 0;
      for(int i = 0; i < t; i++)
        num_0 += val[i], sum_0 += i*val[i];
      mean_0 = (float)sum_0 / num_0;
      mean_1 = (float)(sum_tol - sum_0) / (num_tol - num_0);
      w_0 = (float)num_0 / num_tol;
      s_inter = w_0 * (1 - w_0) * (mean_0 - mean_1) * (mean_0 - mean_1);
      if(s_inter > s_inter_max) {
        s_inter_max = s_inter;
        bin_threshold_l = t;
        bin_threshold_r = t;
      }
      else if(s_inter == s_inter_max) 
        bin_threshold_r = t;
    }
    return (bin_threshold_l+bin_threshold_r)/2;
  }
};

Mat binarize(const Mat &I, uchar bin_t) {
  Mat T = Mat::zeros(I.rows, I.cols, CV_8U);
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      if(I.at<uchar>(i, j) > bin_t)
        T.at<uchar>(i, j) = 255;
      else 
        T.at<uchar>(i, j) = 0;
    }
  }
  return T;
}

Mat dilate(const Mat &I) {
  int dy[] = { 0, 0, 1, -1 };
  int dx[] = { 1, -1, 0, 0 };
  Mat T = I.clone();
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      if(!I.at<uchar>(i, j))
        continue;
      for(int k = 0; k < 4; k++) {
        if(i+dy[k] < 0 || i+dy[k] >= I.rows || j+dx[k] < 0 || j+dx[k] >= I.cols)
          continue;
        if(!I.at<uchar>(i+dy[k], j+dx[k]))
          T.at<uchar>(i, j) = 0;
      }
    }
  }
  return T;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat I = img.clone();
  // 灰度化
  cvtColor(I, I, COLOR_BGR2GRAY);
  // 生成灰度直方图
  grayHist v(256);
  v.makeHist(I);
  // 利用 OtsuMethond 进行二值化
  Mat A = binarize(I, v.OstuMethod());
  // 膨胀 Dilate
  Mat B = dilate(A);
  // 显示图像
  imshow("before", I);
  imshow("Otsu", A);
  imshow("dilate", B);
  waitKey();
  return 0;
}