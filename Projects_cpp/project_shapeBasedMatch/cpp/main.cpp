#include "line2d.hpp"
using namespace cv;
using namespace std;

int main() {
  Mat image = imread("../imagelib/Template.jpg", IMREAD_COLOR);
  // Ptr<shapeInfo_producer> sip = shapeInfo_producer::load_config(image);
  Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(image);
  sip->produce_infos();
  for(const auto &info : sip->Infos_constptr()) {
    cout << "angle: " << info.angle << endl;
    cout << "scale: " << info.scale << endl;
    Mat templateImage;
    sip->src_of(info).copyTo(templateImage, sip->mask_of(info));
    Ptr<Template> tp = Template::create_from(templateImage, 100, 0.2);
    for (const auto &pg : tp->prograds) {
      circle(templateImage, pg.p_xy, 1, Scalar(0, 0, 255), -1);
    }

    namedWindow("templateImage", WINDOW_NORMAL);
    imshow("templateImage", templateImage);
    waitKey();
    break;
  }
  waitKey();
  return 0;
}
