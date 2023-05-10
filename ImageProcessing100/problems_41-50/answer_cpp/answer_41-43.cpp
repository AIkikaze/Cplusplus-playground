/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-10 08:54:37
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-10 17:42:36
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_41-50\answer_cpp\answer_41-43.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
using namespace cv;

Mat gaussianFilter(const Mat &I, Size k_size, double Sigma) {
  // 变常量声明、初始化
  Mat T = Mat::zeros(I.rows, I.cols, CV_64F);
  double kernel_sum = 0.0;
  vector<vector<double>> kernel(k_size.height, vector<double>(k_size.width, 0.0));
  vector<int> dy, dx;
  // 初始化高斯滤波器和位移增量 
  for(int i = 0; i < k_size.height; i++)
    dy.push_back(-(k_size.height>>1)+i);
  for(int i = 0; i < k_size.width; i++)
    dx.push_back(-(k_size.width>>1)+i);
  for(int i = 0; i < k_size.height; i++) {
    for(int j = 0; j < k_size.width; j++) {
      double xy2 = (double)dy[i]*dy[i] + (double)dx[i]*dx[i];
      kernel[i][j] = exp( - xy2 / (2*Sigma*Sigma) ) / ( 2 * CV_PI * Sigma * Sigma );
      kernel_sum += kernel[i][j];
    }
  }
  // 归一化高斯核
  for(int i = 0; i < k_size.height; i++) {
    for(int j = 0; j < k_size.width; j++) {
      kernel[i][j] /= kernel_sum;
    }
  }
  // padding 滤波计算
  double val;
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      val = 0.0;
      for(int y = 0; y < k_size.height; y++) {
        for(int x = 0; x < k_size.width; x++) {
          if(i+dy[y] < 0 || i+dy[y] >= I.rows || j+dx[x] < 0 || j+dx[x] >= I.cols)
            continue;
          val += I.at<double>(i+dy[y], j+dx[x]) * kernel[y][x];
        }
      }
      T.at<double>(i, j) = val;
    }
  }
  return T;
}

// 水平方向 sobel 算子
// vecx^T vecy =  1 0 -1 
//              [ 2 0 -2 ]
//                1 0 -1
// 垂直方向 sobel 算子
// vecy^T vecx =  1 2 1
//              [ 0 0 0 ]
//               -1-2-1
Mat sobelFilter(const Mat &I) {
  // 常量初始化
  Mat T = Mat::zeros(I.rows, I.cols, CV_64FC2);
  int dyx[] = { -1, 0, 1 };
  int vecx[] = { 1, 2, 1 };
  int vecy[] = { 1, 0, -1 };
  // pading 滤波计算
  double valy, valx;
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      valy = valx = 0.0;
      for(int y = 0; y < 3; y++) {
        for(int x = 0; x < 3; x++) {
          if(i+dyx[y] < 0 || i+dyx[y] >= I.rows || j+dyx[x] < 0 || j+dyx[x] >= I.cols)
            continue;
          // 垂直 y 方向
          valy += I.at<double>(i+dyx[y], j+dyx[x]) * vecx[x] * vecy[y];
          // 水平 x 方向
          valx += I.at<double>(i+dyx[y], j+dyx[x]) * vecx[y] * vecy[x];
        }
      }
      T.at<Vec2d>(i, j)[0] = valy;
      T.at<Vec2d>(i, j)[1] = valx;
    }
  }
  return T;
}

// 对经过 sobel 滤波的差分矩阵进行量化绘图
void gradPlot(const Mat &I) {
  // 初始化 梯度矩阵 和 方向角矩阵
  Mat edge = Mat::zeros(I.rows, I.cols, CV_64F);
  Mat angle = Mat::zeros(I.rows, I.cols, CV_8U);
  // 梯度和方向角计算
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      edge.at<double>(i, j) = sqrt(pow(I.at<Vec2d>(i, j)[0], 2) + pow(I.at<Vec2d>(i, j)[1], 2));
      double a_tan = atan(I.at<Vec2d>(i, j)[0] / I.at<Vec2d>(i, j)[1]);
      if(a_tan > -0.4142 && a_tan <= 0.4142)
        angle.at<uchar>(i, j) = 0;
      else if(a_tan > 0.4142 && a_tan < 2.4142)
        angle.at<uchar>(i, j) = 45;
      else if(abs(a_tan) >= 2.4142)
        angle.at<uchar>(i, j) = 90;
      else if(a_tan > -2.4142 && a_tan <= -0.4142)
        angle.at<uchar>(i, j) = 135;
    }
  }
  // 非极大值抑制 
  copyMakeBorder(edge, edge, 1, 1, 1, 1, BORDER_CONSTANT, 0.0);
  for(int i = 1; i < edge.rows-1; i++) {
    for(int j = 1; j < edge.cols-1; j++) {
      if(angle.at<uchar>(i, j) == 0) {
        if(edge.at<double>(i, j) < edge.at<double>(i, j+1) || edge.at<double>(i, j) < edge.at<double>(i, j+1))
          edge.at<double>(i, j) = 0;
      }
      else if(angle.at<uchar>(i, j) == 45) {
        if(edge.at<double>(i, j) < edge.at<double>(i-1, j-1) || edge.at<double>(i, j) < edge.at<double>(i+1, j+1))
          edge.at<double>(i, j) = 0;
      }
      else if(angle.at<uchar>(i, j) == 90) {
        if(edge.at<double>(i, j) < edge.at<double>(i-1, j) || edge.at<double>(i, j) < edge.at<double>(i+1, j))
          edge.at<double>(i, j) = 0;
      }
      else if(angle.at<uchar>(i, j) == 135){
        if(edge.at<double>(i, j) < edge.at<double>(i+1, j-1) || edge.at<double>(i, j) < edge.at<double>(i+1, j-1))
          edge.at<double>(i, j) = 0;
      }
    }
  }
  edge = edge(Range(1, edge.rows-1), Range(1, edge.cols-1));
  // 格式转化
  edge.convertTo(edge, CV_8U);
  // normalize(edge, edge, 0, 255, NORM_MINMAX, CV_8U);
  // 将 edge 二值化
  // for(int i = 0; i < edge.rows; i++) {
  //   for(int j = 0; j < edge.cols; j++) {
  //     if(edge.at<uchar>(i, j) < 20) 
  //       edge.at<uchar>(i, j) = 0;
  //     if(edge.at<uchar>(i, j) >  100) 
  //       edge.at<uchar>(i, j) = 255;
  //   }
  // }
  // for(int i = 0; i < edge.rows; i++) {
  //   for(int j = 0; j < edge.cols; j++) {
  //     if(edge.at<uchar>(i, j) > 0 && edge.at<uchar>(i, j) < 255) {
  //       for(int di = -1; di < 2; di++) {
  //         for(int dj = -1; dj < 2; dj++) {
  //           if(edge.at<uchar>(i, j) == 255)
  //             continue;
  //           if(edge.at<uchar>(i+di, j+dj) == 255) 
  //             edge.at<uchar>(i, j) = 255;
  //         }
  //       }
  //     }
  //   }
  // }
  // 显示图像
  imshow("edge", edge);
  imshow("angle", angle);
}

int main() {
  // 读取图像
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat I = img.clone();
  // 对图像进行灰度化处理
  cvtColor(I, I, COLOR_BGR2GRAY);
  // 格式转换
  I.convertTo(I, CV_64F);
  // 高斯滤波
  Mat A = gaussianFilter(I, Size(5, 5), 1.4);
  Mat B = sobelFilter(A);
  // 显示图像
  imshow("img", img);
  gradPlot(B);
  waitKey();
  return 0;
}