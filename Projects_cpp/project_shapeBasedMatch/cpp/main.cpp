#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;

void template_test() {
  Mat image = imread("../imagelib/Template.jpg", IMREAD_COLOR);

  Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(image);
  sip->produce_infos();

  Template::TemplateParams params;
  params.num_features = 50;
  Ptr<Template> tp = makePtr<Template>(params);

  for (const auto &info : sip->Infos_constptr()) {
    cout << "angle: " << info.angle << endl;
    cout << "scale: " << info.scale << endl;

    Mat templateImage;
    sip->src_of(info).copyTo(templateImage, sip->mask_of(info));

    if (!tp->iscreated())
      tp->create_from(templateImage);
    vector<Template::Features> featurePoints = tp->relocate_by(info);

    Point center(templateImage.cols / 2, templateImage.rows / 2);
    for (const auto &p : featurePoints) {
      circle(templateImage, center + p.p_xy, 1, Scalar(0, 0, 255), -1);
    }

    namedWindow("templateImage", WINDOW_NORMAL);
    imshow("templateImage", templateImage);
    waitKey();
  }
}

int main() {
  // template_test();

  Mat sourceImage = imread("../imagelib/Template.jpg", IMREAD_GRAYSCALE);
  Mat templateImage = imread("../imagelib/Template.jpg", IMREAD_COLOR);

  Template::TemplateParams params;
  params.num_features = 200;
  params.nms_kernel_size = 3;
  Detector detector;
  detector.match(sourceImage, templateImage, 90, params);

  return 0;
}
