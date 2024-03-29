#include "line2dup.hpp"
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
  Mat sourceImage = imread("../../imagelib/source_0.bmp", IMREAD_COLOR);
  Mat templateImage = imread("../../imagelib/template_0.bmp", IMREAD_COLOR);

  line2Dup::Detector detector;
  detector.match(sourceImage, templateImage, 90);

  vector<Vec6f> points;
  vector<RotatedRect> boxes;
  detector.detectBestMatch(points, boxes);

  return 0;
}
