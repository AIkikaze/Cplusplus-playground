#include "line2d.hpp"
using namespace cv;
using namespace std;

int main() {
  Mat image = imread("../imagelib/Template.jpg", IMREAD_COLOR);
  Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(image);
  sip->produce_infos();
  for(const auto &info : sip->Infos_constptr()) {
    cout << "angle: " << info.angle << endl;
    cout << "scale: " << info.scale << endl;
    Mat templateImage;
    sip->src_of(info).copyTo(templateImage, sip->mask_of(info));
    Template::TemplateParams params;
    params.num_features = 100;
    Ptr<Template> tp = Template::makePtr_from(templateImage, params);
    for (const auto &pg : tp->pg_ptr()) {
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
