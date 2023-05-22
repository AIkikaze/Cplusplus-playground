#include "GeoMatch.hpp"
using namespace std;
using namespace cv;

int main() {
  Mat tempImage = imread("../imagelib/mount.png", IMREAD_COLOR);
  Mat sourceImage = imread("../imagelib/mounts.png", IMREAD_COLOR);
  GeoMatch gm;
  gm.setSourceImage(sourceImage);
  gm.setTempImage(tempImage);
  gm.processImage();
  gm.createGeoMatchModel(50, 150);

  Mat ScoreMap;
  ScoreMap = gm.getScoreMap();

  imshow("ScoreMap", ScoreMap);

  waitKey();
  return 0;
}