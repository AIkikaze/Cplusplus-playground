/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-08 15:34:54
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 13:42:49
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_37.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;

// T = 8
// F(u,v) = 2 / T * C(u)C(v) * Sum_{y=0:T-1} Sum_{x=0:T-1} 
//                  f(y,x) cos((2y+1)u*pi/2T) cos((2x+1)v*pi/2T)
Mat dct(const Mat &I, int T) {
  double val, Cu, Cv;
  double cos1, cos2;
  double fator = CV_PI / (2.0 * T);
  Mat f = Mat::zeros(I.rows, I.cols, CV_64F);

  for(int r = 0; r < I.rows; r += T) {
    for(int l = 0; l < I.cols; l += T) {
      for(int u = 0; u < T; u++) {
        for(int v = 0; v < T; v++) {
          val = 0.0;
          Cu = Cv = 1.0;
          if(!u) Cu = 1. / sqrt(2);
          if(!v) Cv = 1. / sqrt(2);
          for(int y = 0; y < T; y++) {
            for(int x = 0; x < T; x++) {
              cos1 = cos((double)(2*y+1) * u * fator);
              cos2 = cos((double)(2*x+1) * v * fator);
              val += Cu * Cv * (double)I.at<uchar>(r+y, l+x) * cos1 * cos2;
            }
          }
          val *= 2.0 / T;
          f.at<double>(r+u, l+v) = val;
        }
      }
    }
  }

  return f;
}

// f(x,y) = 2 / T * Sum_{u=0:T-1} Sum_{v=0:T-1} * C(u)C(v) 
//                  F(u,v) cos((2x+1)u*pi/2T) cos((2y+1)v*pi/2T)
Mat idct(const Mat &f, int T, int K) {
  double val, Cu, Cv;
  double cos1, cos2;
  double fator = CV_PI / (2.0 * T);
  Mat I = Mat::zeros(f.rows, f.cols, CV_8U);

  for(int r = 0; r < I.rows; r += T) {
    for(int l = 0; l < I.cols; l += T) {
      for(int y = 0; y < T; y++) {
        for(int x = 0; x < T; x++) {
          val = 0.0;
          for(int u = 0; u < K; u++) {
            for(int v = 0; v < K; v++) {
              Cu = Cv = 1.0;
              if(!u) Cu = 1. / sqrt(2);
              if(!v) Cv = 1. / sqrt(2);
              cos1 = cos((double)(2*y+1) * u * fator);
              cos2 = cos((double)(2*x+1) * v * fator);
              val += Cu * Cv * f.at<double>(r+u, l+v) * cos1 * cos2;
            }
          }
          val *= 2.0 / T;
          val = val > 255 ? 255 : fmax(val, 0);
          I.at<uchar>(r+y, l+x) = uchar(val);
        }
      }
    }
  }

  return I;
}

// 求峰值信噪比 (Peak Signal to Noise Ratio) 
double getPSNR(const Mat &I, const Mat &F) {
  long val = 0;
  double v_max = 255.0;
  double mse = 0.0;
  double psnr = 0.0;

  for(int y = 0; y < I.rows; y++) {
    for(int x = 0; x < I.cols; x++) {
      double tmp = (double)(I.at<uchar>(y, x)-F.at<uchar>(y, x));
      val += pow(tmp, 2);
    }
  }
  mse = (double)val / (I.rows * I.cols);
  psnr = 10 * log10(v_max * v_max / mse);

  return psnr;
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_GRAYSCALE);
  int T = 8;
  int K = 4;

  Mat A = dct(img, T);
  Mat B = idct(A, T, K);
  double psnr = getPSNR(img, B);
  double bit_rate = 8.0 * pow((double)K, 2) / pow((double)T, 2);
  
  cout << "psnr:" << psnr << endl;
  cout << "bit_rate:" << bit_rate << endl;
  imshow("origin", img);
  imshow("dct+idct", B);
  waitKey();
  return 0;
}