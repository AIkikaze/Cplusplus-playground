/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-08 15:34:54
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-09 15:45:18
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_31-40\answer_cpp\answer_40.cpp
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

void rgb2YCbCr(Mat &I) {
  double Y, Cb, Cr;
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Vec3d p = I.at<Vec3d>(i, j);
      Y = 0.299 * p[2] + 0.5870 * p[1] + 0.114 * p[0];
      Cb = - 0.1687 * p[2] - 0.3323 * p[1] + 0.5 * p[0] + 128;
      Cr = 0.5 * p[2] - 0.4187 * p[1] - 0.0813 * p[0] + 128;
      I.at<Vec3d>(i, j)[0] = Y;
      I.at<Vec3d>(i, j)[1] = Cb;
      I.at<Vec3d>(i, j)[2] = Cr;
    }
  }
}

void ycbcr2RGB(Mat &I) {
  double R, G, B;
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      Vec3d p = I.at<Vec3d>(i, j);
      R = p[0] + (p[2] - 128) * 1.4102;
      G = p[0] - (p[1] - 128) * 0.3441 - (p[2] - 128) * 0.7139;
      B = p[0] + (p[1] - 128) * 1.7718;
      I.at<Vec3d>(i, j)[0] = B;
      I.at<Vec3d>(i, j)[1] = G;
      I.at<Vec3d>(i, j)[2] = R;    
    }
  }
}

// T = 8
// F(u,v) = 2 / T * C(u)C(pv) * Sum_{y=0:T-1} Sum_{x=0:T-1} 
//                  f(y,x) cos((2y+1)u*pi/2T) cos((2x+1)v*i/2T)
Mat dct(const Mat &I, int T) {
  double val, Cu, Cv;
  double cos1, cos2;
  double fator = CV_PI / (2.0 * T);
  Mat f = Mat::zeros(I.rows, I.cols, CV_64FC3);

  for(int r = 0; r < I.rows; r += T) {
    for(int l = 0; l < I.cols; l += T) {
      for(int c = 0; c < I.channels(); c++) {
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
                val += Cu * Cv * I.at<Vec3d>(r+y, l+x)[c] * cos1 * cos2;
              }
            }
            val *= 2.0 / T;
            f.at<Vec3d>(r+u, l+v)[c] = val;
          }
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
  Mat I = Mat::zeros(f.rows, f.cols, CV_64FC3);

  for(int r = 0; r < I.rows; r += T) {
    for(int l = 0; l < I.cols; l += T) {
      for(int c = 0; c < 3; c++) {
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
                val += Cu * Cv * f.at<Vec3d>(r+u, l+v)[c] * cos1 * cos2;
              }
            }
            val *= 2.0 / T;
            I.at<Vec3d>(r+y, l+x)[c] = val;
          }
        }
      }
    }
  }

  return I;
}

void quantize(Mat &I, int T) {
  double Q1[T][T] = {{16, 11, 10, 16, 24, 40, 51, 61},
                    {12, 12, 14, 19, 26, 58, 60, 55},
                    {12, 12, 14, 19, 26, 58, 60, 55},
                    {14, 17, 22, 29, 51, 87, 80, 62},
                    {18, 22, 37, 56, 68, 109, 103, 77},
                    {24, 35, 55, 64, 81, 104, 113, 92},
                    {49, 64, 78, 87, 103, 121, 120, 101},
                    {72, 92, 95, 98, 112, 100, 103, 99}
                    };
  double Q2[T][T] = {{17, 18, 24, 47, 99, 99, 99, 99},
                    {18, 21, 26, 66, 99, 99, 99, 99},
                    {24, 26, 56, 99, 99, 99, 99, 99},
                    {47, 66, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99}
                    };

  for(int r = 0; r < I.rows; r += T) {
    for(int l = 0; l < I.cols; l += T) {
      for(int y = 0; y < T; y++) {
        for(int x = 0; x < T; x++) {
          I.at<Vec3d>(y, x)[0] = round(I.at<Vec3d>(y, x)[0] / Q1[y][x]) * Q1[y][x];
          I.at<Vec3d>(y, x)[1] = round(I.at<Vec3d>(y, x)[1] / Q2[y][x]) * Q2[y][x];
          I.at<Vec3d>(y, x)[2] = round(I.at<Vec3d>(y, x)[2] / Q2[y][x]) * Q2[y][x];
        }
      }
    }
  }
}

int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  int T = 8;
  int K = 7;
  Mat I = img.clone();

  I.convertTo(I, CV_64FC3);
  rgb2YCbCr(I);           
  Mat A = dct(I, T);      
  quantize(A, T);         
  Mat B = idct(A, T, K);  
  ycbcr2RGB(B);           
  B.convertTo(B, CV_8UC3);

  imshow("origin", img);
  imwrite("../imagelib/origin.jpg", img);
  imshow("jpeg", B);
  imwrite("../imagelib/jpeg_compress.jpg", B);
  waitKey();
  return 0;
}