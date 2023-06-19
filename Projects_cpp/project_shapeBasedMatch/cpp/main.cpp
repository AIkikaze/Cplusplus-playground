#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;
extern double __time__relocate__;
extern double __time__produceroi__;

void template_test() {
  Mat image = imread("../imagelib/rotate_base.jpg", IMREAD_COLOR);

  Ptr<shapeInfo_producer> sip =  makePtr<shapeInfo_producer>(image);
  sip->angle_range = {0, 360};
  sip->angle_step = 1.0;
  sip->scale_range = {0.8, 1.2};
  sip->scale_step = 0.1;
  sip->produce_infos();

  Template::TemplateParams params;
  params.num_features = 500;
  Ptr<Template> tp = makePtr<Template>(params);

  for (const auto &info : sip->Infos_constptr()) {
    cout << "angle: " << info.angle << endl;
    cout << "scale: " << info.scale << endl;

    namedWindow("srcImage", WINDOW_NORMAL);
    imshow("srcImage", sip->src_of(info));
    namedWindow("maskImage", WINDOW_NORMAL);
    imshow("maskImage", sip->mask_of(info));
    waitKey();
    destroyAllWindows();

    Mat templateImage;
    sip->src_of(info).copyTo(templateImage, sip->mask_of(info));

    if (!tp->iscreated())
      tp->create_from(templateImage);

    vector<Template::Feature> featurePoints = tp->relocate_by(info);

    Point center(templateImage.cols / 2, templateImage.rows / 2);
    for (const auto &p : featurePoints) {
      drawMarker(templateImage, center + p.p_xy, Scalar(0, 0, 255), MARKER_SQUARE, 10);
    }

    namedWindow("templateImage", WINDOW_NORMAL);
    imshow("templateImage", templateImage);
    imwrite("templateImage.jpg", templateImage);
    waitKey();
  }
}

int main() {
  // template_test();

  // Mat sourceImage = imread("../imagelib/rotate_1.png", IMREAD_GRAYSCALE);
  // Mat templateImage = imread("../imagelib/base.jpg", IMREAD_GRAYSCALE);

  // pyrDown(sourceImage, sourceImage);
  // pyrDown(templateImage, templateImage);

  // adaptiveThreshold(sourceImage, sourceImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 53, 2.0);
  // adaptiveThreshold(templateImage, templateImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 53, 2.0);
  // cv::threshold(templateImage, templateImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
 
  Mat sourceImage = imread("../imagelib/template_0.bmp", IMREAD_COLOR);
  Mat templateImage = imread("../imagelib/template_0.bmp", IMREAD_COLOR);

  Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(templateImage);
  sip->angle_range = {0, 360};
  sip->angle_step = 1.0;
  sip->scale_range = {0.8, 1.0};
  sip->scale_step = 0.1;
  sip->produce_infos();

  Template::TemplateParams params;
  params.num_features = 100;
  params.nms_kernel_size = 3;
  params.scatter_distance = 12.0f;

  Timer time;
  Detector detector;
  detector.match(sourceImage, templateImage, 80, params);

  vector<Vec6f> points;
  detector.detectBestMatch(points);
  for (int i = 0; i < (int)points.size(); i++) {
    printf("match point [%3d] : \n x -> %5f \n y -> %5f \n scale -> %5f \n angle -> %5f \n score -> %5f \n",
            i, points[i][0], points[i][1], 
            points[i][2], points[i][3],
            points[i][4]);
  }

  time.out("模板匹配运行完毕!");
  cout << "特征点旋转运算时间: " << __time__relocate__ << endl;
  cout << "联通域分析运算时间: " << __time__produceroi__ << endl;

  detector.draw();

  return 0;
}