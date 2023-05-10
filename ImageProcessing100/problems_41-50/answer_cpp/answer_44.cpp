/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-10 14:10:36
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-10 17:12:06
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_41-50\answer_cpp\answer_44.cpp
 * @Description: 
 * 
 */
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <set>
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
Mat getEdge(const Mat &I) {
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
        if(edge.at<double>(i, j) < edge.at<double>(i, j-1) || edge.at<double>(i, j) < edge.at<double>(i, j+1))
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
        if(edge.at<double>(i, j) < edge.at<double>(i-1, j+1) || edge.at<double>(i, j) < edge.at<double>(i+1, j-1))
          edge.at<double>(i, j) = 0;
      }
    }
  }
  edge = edge(Range(1, edge.rows-1), Range(1, edge.cols-1));
  // 格式转化
  normalize(edge, edge, 0, 255, NORM_MINMAX, CV_8U);
  // 将 edge 二值化
  for(int i = 0; i < edge.rows; i++) {
    for(int j = 0; j < edge.cols; j++) {
      if(edge.at<uchar>(i, j) < 30) 
        edge.at<uchar>(i, j) = 0;
      if(edge.at<uchar>(i, j) > 100) 
        edge.at<uchar>(i, j) = 255;
    }
  }
  for(int i = 0; i < edge.rows; i++) {
    for(int j = 0; j < edge.cols; j++) {
      if(edge.at<uchar>(i, j) > 0) {
        for(int di = -1; di < 2; di++)
          for(int dj = -1; dj < 2; dj++)
            if(!di && !dj)
              continue;
            else if(edge.at<uchar>(i, j) < 255 || edge.at<uchar>(i+di, j+dj) == 255)
              edge.at<uchar>(i, j) = 255;
      }
    }
  }
  return edge;
}

Mat houghTrans(const Mat &I) {
  double rou, angle;
  double r_max = sqrt(I.rows*I.rows + I.cols*I.cols);
  Mat T = Mat::zeros((int)r_max*2, 180, CV_8U);
  // 计算票数统计
  for(int y = 0; y < I.rows; y++) {
    for(int x = 0; x < I.cols; x++) {
      if(!I.at<uchar>(y, x))
        continue;
      for(int t = 0; t < 180; t++) {
        angle = CV_PI * ((double)t/180.0);
        rou = (double)x*cos(angle) + (double)y*sin(angle) + r_max;
        // cout << "[" << y << "," << x << "]:";
        // cout << (int)rou << " " << t << endl; cin.get();
        T.at<uchar>((int)rou, t)++;
      }
    }
  }
  return T;
}

struct Elemt {
  int val, y, x;
  Elemt(int v, int a, int b): val(v), y(a), x(b) {}
  bool operator<(const Elemt& other) const {
        return val < other.val;
  }
};

vector<Elemt> getNMS(const Mat &I, unsigned long long st_size) {
  vector<Elemt> result;
  multiset<Elemt> st;
  Mat T = I.clone();
  for(int i = 0; i < I.rows; i++) {
    for(int j = 0; j < I.cols; j++) {
      for(int u = -1; u < 2; u++) {
        for(int v = -1; v < 2; v++) {
          if(!u && !v)
            continue;
          if(T.at<uchar>(i+u, j+v) > T.at<uchar>(i, j)) {
            T.at<uchar>(i, j) = 0;
            break;
          }
        }
      }
      if(!T.at<uchar>(i, j)) 
        continue;
      st.insert(Elemt(T.at<uchar>(i, j), i, j));
      if(st.size() > st_size)
        st.erase(st.begin()); // 移除最小元素
    }
  }
  T.setTo(0);
  while(st.size() > 0) {
    Elemt tmp = (*st.begin());
    T.at<uchar>(tmp.y, tmp.x) = tmp.val;
    result.push_back(tmp);
    st.erase(st.begin());
  }
  imshow("NMS", T);
  return result;
}

Mat houghInvTrans(const Mat &I, vector<Elemt> p_list) {
  double r_max = sqrt(I.rows*I.rows + I.cols*I.cols);
  Mat T = I.clone();
  // i:val->vote number; y->rou ; x->t
  for(const auto& i : p_list) {
    for(int x = 0; x < I.cols; x++) {
      double theta = CV_PI * ((double)i.x/180.0);
      int y = - cos(theta) / sin(theta) * x + (i.y-r_max) / sin(theta);
      if(y >= 0 && y < I.rows)
        T.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
    }
    for(int y = 0; y < I.rows; y++) {
      double theta = CV_PI * ((double)i.x/180.0);
      int x = sin(theta) / cos(theta) * y + (i.y-r_max) / cos(theta);
      if(x >= 0 && x < I.cols)
        T.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
    }
  }
  return T;
}

int main() {
  Mat img = imread("../imagelib/thorino.jpg", IMREAD_COLOR);
  Mat I = img.clone();
  // 灰度化
  cvtColor(I, I, COLOR_BGR2GRAY);
  // 格式转换
  I.convertTo(I, CV_64F);
  // 高斯滤波
  Mat A = gaussianFilter(I, Size(5, 5), 1.4);
  // Sobel 滤波
  Mat B = sobelFilter(A);
  // 梯度量化
  Mat C = getEdge(B);
  // Hough 变换
  Mat D = houghTrans(C);
  // NMS 求局部最大值
  vector<Elemt> E = getNMS(D, 10);
  // Hough 逆变换
  Mat F = houghInvTrans(img, E);
  // 显示图像
  imshow("img", img);
  imshow("Hough", F);
  waitKey();
  return 0;
}