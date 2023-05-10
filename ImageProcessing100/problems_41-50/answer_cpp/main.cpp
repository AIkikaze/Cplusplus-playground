/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-10 10:43:51
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-10 10:43:59
 * @FilePath: \Cplusplus-playground\ImageProcessing100\problems_41-50\answer_cpp\main.cpp
 * @Description: 
 * 
 */
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat mat64f = Mat::ones(3, 3, CV_64F);
    Mat mat8u;
    mat64f.convertTo(mat8u, CV_8U);
    cout << "mat8u = " << endl << mat8u << endl; // 输出截断后的数据
    normalize(mat64f, mat8u, 0, 255, NORM_MINMAX, CV_8U);
    cout << "normalized mat8u = " << endl << mat8u << endl; // 输出归一化后的数据
    cin.get();
    return 0;
}
