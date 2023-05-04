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

// function: I 中对每个像素交换色彩通道 BRR2RGB
// input: 格式为 CV_8U3C 的 Mat 像素矩阵 I
// ouput: 像素矩阵 T
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
