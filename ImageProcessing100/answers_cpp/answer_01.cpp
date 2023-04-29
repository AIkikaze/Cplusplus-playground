#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;

Mat gray(Mat &I) {
    Mat T = I.clone();
    return T;
}

int main() {
    // 载入图片
    Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
    Mat A = gray(img);
    imshow("hello", img); // 在窗口 "hello" 中显示图片
    waitKey();            // 等待用户按下键盘
    return 0;
}
