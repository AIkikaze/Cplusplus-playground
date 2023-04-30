/*
author: wenzy
modified date: 20230429
target: design an algorithm to change the color channdels of an image from BGR to RGB 
*/
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
    Mat A = channelSwap(img);
    imshow("problem_01-channelSwap", A); // 在窗口 "hello" 中显示图片
    waitKey();            // 等待用户按下键盘
    return 0;
}
