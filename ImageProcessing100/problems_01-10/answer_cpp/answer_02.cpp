/*
author: wenzy
modified date: 20230429
target: design an algorithm to transform a colored image to a Grayscaled image
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat grayScale(Mat &I) {
  int nRows = I.rows;
  int nCols = I.cols;
  Mat T = Mat::zeros(nRows, nCols, CV_8U);
  for(int i = 0; i < nRows; i++)
    for(int j = 0; j < nCols; j++) {
      uchar b = I.at<Vec3b>(i, j)[0];
      uchar g = I.at<Vec3b>(i, j)[1];
      uchar r = I.at<Vec3b>(i, j)[2];
      T.at<uchar>(i, j) = (uchar)( 0.2126 * b + 0.7152 * g + 0.0722 * r );
    }
  return T;
}

int main() {
  Mat img = imread("../imagelib/test.jpeg", IMREAD_COLOR);
  Mat A = grayScale(img);
  imwrite("../imagelib/test_grayScale.jpg", A);
  imshow("grayScale", A);
  waitKey();
  return 0;
}