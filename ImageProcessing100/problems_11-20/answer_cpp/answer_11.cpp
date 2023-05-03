/*
author: wenzy
modified date: 20230502
*/
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;
using namespace cv;

Mat smoothFilter(Mat &I, Size S) {
    int n_row = I.rows;
    int n_col = I.cols;
    int n_channel = I.channels();
    int n_wid = S.width;
    int n_hei = S.height;
    unique_ptr<int[]> dx(new int[n_wid * n_hei]);
    unique_ptr<int[]> dy(new int[n_wid * n_hei]);
    Mat T = Mat::zeros(n_row, n_col, CV_8UC3);

    // 初始化移增量
    for (int i = 0; i < n_hei; i++)
        for (int j = 0; j < n_wid; j++) {
            int _idx = i * n_wid + j;
            dy[_idx] = -(n_wid >> 1) + j;
            dx[_idx] = -(n_hei >> 1) + i;
        }

    // padding 滤波计算
    for (int i = 0; i < n_row; i++)
        for (int j = 0; j < n_col; j++)
            for (int c = 0; c < n_channel; c++) {
                int _p_sum = 0;
                for (int _i = 0; _i < n_wid * n_hei; _i++) {
                    if (i + dx[_i] < 0 || i + dx[_i] >= n_row ||
                        j + dy[_i] < 0 || j + dy[_i] >= n_col)
                        continue;
                    _p_sum += I.at<Vec3b>(i + dx[_i], j + dy[_i])[c];
                }
                T.at<Vec3b>(i, j)[c] = _p_sum / (n_wid * n_hei);
            }

    return T;
}

int main() {
    Mat img = imread("../imagelib/imori_noise.jpg", IMREAD_COLOR);
    Mat A = smoothFilter(img, Size(21, 21));
    imshow("before", img);
    imshow("smoothFilter", A);
    waitKey();
    return 0;
}
