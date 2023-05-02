/*
author: wenzy
modified date: 20230501
target: divide the grid into 8x8 and average pool
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
using namespace cv;
using namespace std;

inline
void averagePoolPixel(Mat &T, Mat &I, int ROW, int COL, int STEP) {
  int sum_tol[] = {0, 0, 0};
  for(int i = ROW; i < ROW + 8; i++)
    for(int j = COL; j < COL + 8; j++)
      for(int c = 0; c < 3; c++)
        sum_tol[c] += I.at<Vec3b>(i, j)[c];
  
  for(int i = ROW; i < ROW + 8; i++)
    for(int j = COL; j < COL + 8; j++)
      for(int c = 0; c < 3; c++)
        T.at<Vec3b>(i, j)[c] = (float) sum_tol[c] / 64;
}

Mat averagePool(Mat &I, int STEP) {
  CV_Assert(I.type() == CV_8UC3);

  int n_row = I.rows;
  int n_col = I.cols;
  int row_st = 0;
  int col_st = 0;
  Mat T = Mat::zeros(n_row, n_col, CV_8UC3);
  if(n_row % STEP || n_col % STEP) {
    row_st = (n_row % STEP)>>1;
    col_st = (n_col % STEP)>>1;
  }

  for(int i = 0; i < n_row / STEP; i++)
    for(int j = 0; j < n_col / STEP; j++) {
      int idx_row = row_st + i * STEP;
      int idx_col = col_st + j * STEP;
      averagePoolPixel(T, I, idx_row, idx_col, STEP);
    }

  return T;
}

int main() {
  Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
  Mat A = averagePool(img, 8);
  imshow("averagePool", A);
  waitKey();
  return 0;
}
