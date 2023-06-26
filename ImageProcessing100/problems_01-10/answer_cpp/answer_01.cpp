/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-04 08:11:35
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-08 17:14:33
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_01-10\answer_cpp\answer_01.cpp
 * @Description: 
 * 读取图像, 将 RGB 通道的图片转换为 BRG 通道
 */

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
using namespace cv;
using namespace std;

Mat channelSwap(Mat &I) {
    Mat T = I.clone();
    int nRows = I.rows;
    int nCols = I.cols;
    for(int i = 0; i < nRows; i++)
        for(int j = 0; j < nCols; j++) 
            swap(T.at<Vec3b>(i, j)[0], T.at<Vec3b>(i, j)[2]);
    return T;
}

int main() {
    // 载入图片
    Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
    // 调用函数
    Mat A = channelSwap(img);
    // 写入图片
    imwrite("../imagelib/test_channelSwap.jpg", A);
    // 在窗口 "hello" 中显示图片
    imshow("problem_01-channelSwap", A);     
    // 等待用户按下键盘
    waitKey();            
    return 0;
}
