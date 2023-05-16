/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-16 12:07:44
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-16 12:08:59
 * @FilePath: /C++-playground/project_01/cpp/circle.cpp
 * @Description: 
 * 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    Mat image = imread("../imagelib/test.TIFF", IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "Failed to read image!" << std::endl;
        return -1;
    }

    GaussianBlur(image, A, Size(13, 13), 5.4)

    Mat B;
    Canny(A, B, 50, 150);

    std::vector<Vec3f> circles;
    HoughCircles(image, circles, HOUGH_GRADIENT, 1, image.rows / 8, 200, 100, 20, 300);

    Mat outputImage;
    cvtColor(image, outputImage, COLOR_GRAY2BGR);
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cout << center << " radius:" << radius << endl;
        cin.get();
        circle(outputImage, center, radius, Scalar(0, 0, 255), 6);
    }

    imshow("gaussianBlur", A);
    imshow("Canny", B);
    imshow("Circle Detection", outputImage);
    waitKey(0);

    return 0;
}
