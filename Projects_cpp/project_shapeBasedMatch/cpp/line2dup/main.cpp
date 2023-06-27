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

int main() {

  MIPP_test();

  Mat sourceImage = imread("../../imagelib/source_1.png", IMREAD_COLOR);
  Mat templateImage = imread("../../imagelib/template_1.bmp", IMREAD_COLOR);

  // Mat sourceImage = imread("../../imagelib/rotate_0.png", IMREAD_COLOR);
  // Mat templateImage = imread("../../imagelib/rotate_base.jpg", IMREAD_COLOR);


  Timer time;

  typedef line2Dup::ColorGradientPyramid MODAL;

  Ptr<MODAL> modality = makePtr<MODAL>();
  modality->magnitude_threshold = 50.0f;
  modality->count_kernel_size = 3;
  modality->num_features = 100;
  
  line2Dup::Detector detector(3, 4, {3, 3, 3}, modality);
  time.start();
  detector.setSource(sourceImage);
  time.out("__ 源图像初始化完成! __");

  time.start();
  detector.setTemplate(templateImage, cv::Mat(), {0.5, 1.5, 0.025}, {0, 60, 0.25});
  time.out("__ 模板加载完成! __");

  time.start();
  detector.match(50);
  time.out("__ 模板匹配计算完成! __");

  detector.draw(sourceImage);

  vector<Vec6f> points;
  vector<RotatedRect> boxes;
  detector.detectBestMatch(points, boxes);
  for (int i = 0; i < (int)points.size(); i++) {
    printf("match point [%3d] : \n x -> %5f \n y -> %5f \n scale -> %5f \n angle -> %5f \n score -> %5f \n",
            i+1, points[i][0], points[i][1], 
            points[i][2], points[i][3],
            points[i][4]);
  }

  return 0;
}
