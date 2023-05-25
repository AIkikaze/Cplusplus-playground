#include "line-2d.hpp"
using namespace cv;
using namespace std;
using namespace line_2d;

int main() {
  Mat image = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  ImagePyramid IP(image, 5);
  for(int i = 0; i < 5; i++) {
    imshow("imagepyrmid in level" + to_string(i+1), IP[i]);
  }
  waitKey();
  return 0;
}
