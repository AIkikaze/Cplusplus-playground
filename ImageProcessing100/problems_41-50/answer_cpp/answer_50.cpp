#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#define _max(a, b, c) fmax(a, fmax(b, c))
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

Mat erode(const Mat &I) {
  int dy[] = { 0, 0, 1, -1 };
  int dx[] = { 1, -1, 0, 0 };
  Mat T = I.clone();
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      if(I.at<uchar>(i, j))
        continue;
      for(int k = 0; k < 4; k++) {
        if(i+dy[k] < 0 || i+dy[k] >= I.rows || j+dx[k] < 0 || j+dx[k] >= I.cols)
          continue;
        if(I.at<uchar>(i+dy[k], j+dx[k]) == 255)
          T.at<uchar>(i, j) = 255;
      }
    }
  }
  return T;
}

Mat closing(const Mat &I, int times) {
  Mat T = I.clone();
  while(times > 0) {
    // 腐蚀 Erode
    T = erode(T);
    // 膨胀 Dilate
    T = dilate(T);
    times--;
  }
  return T;
}

/* --- Canny 边缘检测 --- */ 

// 高斯滤波
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
          int u = borderInterpolate(i+dy[y], I.rows, BORDER_REPLICATE);
          int v = borderInterpolate(j+dx[x], I.cols, BORDER_REPLICATE);
          val += I.at<double>(u, v) * kernel[y][x];
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
          int u = borderInterpolate(i+dyx[y], I.rows, BORDER_REPLICATE);
          int v = borderInterpolate(j+dyx[x], I.cols, BORDER_REPLICATE);
          // 计算垂直方向 fy 
          valy += I.at<double>(u, v) * vecy[y] * vecx[x];
          // 计算水平方向 fx
          valx += I.at<double>(u, v) * vecx[y] * vecy[x];
        }
      }
      T.at<Vec2d>(i, j)[0] = valy;
      T.at<Vec2d>(i, j)[1] = valx;
    }
  }
  return T;
}

/*
-22.5: -0.414214
22.5: 0.414214
67.5: 2.41421
112.5: -2.41421
157.5: -0.414214
->0^o \in [-0.414214, 0.414214]
  45^o \in [0.414214, 2.41421]
  90^o \in ,-2.41421]U[2.41421,
  135^o \in [-2.41421, -0.414214]
*/
// 对经过 sobel 滤波的差分矩阵进行量化绘图
Mat gradPlot(const Mat &I, int LT = 60, int HT = 100) {
  // 初始化 梯度矩阵 和 方向角矩阵
  Mat edge = Mat::zeros(I.rows, I.cols, CV_64F);
  Mat angle = Mat::zeros(I.rows, I.cols, CV_8U);
  // 梯度和方向角计算
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      edge.at<double>(i, j) = sqrt(pow(I.at<Vec2d>(i, j)[0], 2) + pow(I.at<Vec2d>(i, j)[1], 2));
      if(edge.at<double>(i, j) < 1.0) {
        angle.at<uchar>(i, j) = 175;
        continue;
      }
      if(!I.at<Vec2d>(i, j)[1]) {
        angle.at<uchar>(i, j) = 90;
        continue;
      }
      double a_tan = atan(I.at<Vec2d>(i, j)[0]/I.at<Vec2d>(i, j)[1]);
      if(abs(a_tan) <= 0.414214)
        angle.at<uchar>(i, j) = 0;
      else if(a_tan > 0.414214 && a_tan < 2.41421)
        angle.at<uchar>(i, j) = 45;
      else if(abs(a_tan) >= 2.41421)
        angle.at<uchar>(i, j) = 90;
      else if(a_tan > -2.41421 && a_tan < -0.414214)
        angle.at<uchar>(i, j) = 135;
      // cout << "[" << i << "," << j << "]" << (int)angle.at<uchar>(i, j) << ":" << I.at<Vec2d>(i, j) << endl, cin.get();
    }
  }
  // 非极大值抑制 
  copyMakeBorder(edge, edge, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(0));
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      int dx[2], dy[2];
      if(angle.at<uchar>(i, j) == 0) {
        dx[0] = -1, dx[1] = 1;
        dy[0] = dy[1] = 0;
      }
      else if(angle.at<uchar>(i, j) == 45) {
        dx[0] = dy[0] = -1;
        dx[1] = dy[1] = 1;
      }
      else if(angle.at<uchar>(i, j) == 90) {
        dy[0] = -1, dy[1] = 1;
        dx[0] = dx[1] = 0;
      }
      else if(angle.at<uchar>(i, j) == 135) {
        dx[0] = dy[1] = 1;
        dx[1] = dy[0] = -1;
      }
      else{
        continue;
      }
      if(_max(edge.at<double>(i+1, j+1), edge.at<double>(i+1+dy[0], j+1+dx[0]), edge.at<double>(i+1+dy[1], j+1+dx[1])) != edge.at<double>(i+1, j+1))
        edge.at<double>(i+1, j+1) = 0;
    }
  }
  edge = edge(Range(1, edge.rows-1), Range(1, edge.cols-1));
  // 格式转化
  edge.convertTo(edge, CV_8U);
  // normalize(edge, edge, 0, 255, NORM_MINMAX, CV_8U);
  // 将 edge 二值化
  for(int i = 0; i < edge.rows; i++) {
    for(int j = 0; j < edge.cols; j++) {
      if(edge.at<uchar>(i, j) < LT) 
        edge.at<uchar>(i, j) = 0;
      if(edge.at<uchar>(i, j) >  HT) 
        edge.at<uchar>(i, j) = 255;
    }
  }
  for(int i = 0; i < edge.rows; i++) {
    for(int j = 0; j < edge.cols; j++) {
      for(int di = -1; di < 2; di++) {
        for(int dj = -1; dj < 2; dj++) {
          if(!edge.at<uchar>(i, j) || edge.at<uchar>(i, j) == 255)
            continue;
          if(edge.at<uchar>(i+di, j+dj) == 255) 
            edge.at<uchar>(i, j) = 255;
        }
      }
      if(edge.at<uchar>(i, j) != 255)
        edge.at<uchar>(i, j) = 0;
    }
  }
  return edge;
}
int main() {
  Mat img = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Mat I = img.clone();
  // 对图像进行灰度化处理
  cvtColor(I, I, COLOR_BGR2GRAY);
  // 格式转换
  I.convertTo(I, CV_64F);
  // 高斯滤波
  Mat A = gaussianFilter(I, Size(3, 3), 1.4);
  // Sobel 滤波
  Mat B = sobelFilter(A);
  // 计算边缘梯度
  Mat C = gradPlot(B, 50, 100);
  // 闭运算 closing
  Mat D = closing(C, 1);
  // 显示图像
  imshow("before", img);
  imshow("edge", C);
  imshow("closing operation", D);
  waitKey();
  return 0;
}