#include "line2d.hpp"
using namespace cv;
using namespace std;

int main() {
  Mat image = imread("../imagelib/imori.jpg", IMREAD_COLOR);
  Ptr<shapeInfo_producer> sip = shapeInfo_producer::load_config(image);
  Detector detector;
  for(const auto &info : sip->Infos_constptr()) {
    cout << "angle: " << info.angle << endl;
    cout << "scale: " << info.scale << endl;
    detector.addTemplate(sip->src_of(info), sip->mask_of(info));
  }
  waitKey();
  return 0;
}
