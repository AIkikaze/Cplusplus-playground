#include "line2dup.hpp"
#include "shapeInfo.hpp"
using namespace std;
using namespace cv;

void MIPP_test() {
  cout << "MIPP tests" << endl;
  cout << "----------" << endl << endl;

  cout << "Instr. type:       " << mipp::InstructionType << endl;
  cout << "Instr. full type:  " << mipp::InstructionFullType << endl;
  cout << "Instr. version:    " << mipp::InstructionVersion << endl;
  cout << "Instr. size:       " << mipp::RegisterSizeBit << " bits" << endl;
  cout << "Instr. lanes:      " << mipp::Lanes << endl;
  cout << "64-bit support:    " << (mipp::Support64Bit ? "yes" : "no") << endl;
  cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no")
       << endl;

#ifndef has_max_int8_t
  cout << "in this SIMD, int8 max is not inplemented by MIPP" << endl;
#endif

#ifndef has_shuff_int8_t
  cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << endl;
#endif

  cout << "----------" << endl << endl;
}

class Timer {
public:
  Timer() : start_(0), time_(0) {}

  void start() { start_ = cv::getTickCount(); }

  void stop() {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time() {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

private:
  int64 start_, time_;
};

int main() {
  // bool show_match_result = true;
  // bool show_timings = false;
  // bool learn_online = false;
  int num_classes = 0;
  // int matching_threshold = 80;

  // int learning_lower_bound = 90;
  // int learning_upper_bound = 95;

  Timer extract_timer;
  Timer match_timer;

  namedWindow("template");

  Ptr<line2Dup::Detector> detector = line2Dup::getDefaultLINE();
  string filename = "line2dup_templates.yml";
  
  Mat sourceImage = imread("../../imagelib/source_0.bmp", IMREAD_COLOR);
  Mat templateImage = imread("../../imagelib/template_0.bmp", IMREAD_COLOR);

  Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(templateImage);
  sip->angle_range = {0, 360};
  sip->angle_step = 1.0;
  sip->scale_range = {0.5, 2.0};
  sip->scale_step = 0.1;
  sip->produce_infos();

  vector<Mat> sources;
  for (const auto &info : sip->Infos_constptr()) {
    sources.push_back(sip->src_of(info));
  }

  string class_id = format("class%d", num_classes);
  Rect bb;
  extract_timer.start();
  int template_id = detector->addTemplate(sources, class_id, Mat(), &bb);
  Mat display = sources[0].clone();
  extract_timer.stop();
  if (template_id != -1) {
    printf("*** Added template (id %d) for new object class %d ***\n", template_id, num_classes);
  }
  
  Point p2(bb.x + bb.width, bb.y + bb.height);
  rectangle(display, p1, p2, Scalar(0, 0, 0), 3);
  rectangle(display, p1, p2, Scalar(0, 255, 0), 1);
  line2Dup::drawFeatures(display, detector->getTemplates(class_id, template_id), p1, 3);
  Rect roi(bb.x - 5, bb.y - 5, bb.width + 15, bb.height + 15);
  display = display(roi);

  imshow("template", display);
  waitKey();
  


  return 0;
}
