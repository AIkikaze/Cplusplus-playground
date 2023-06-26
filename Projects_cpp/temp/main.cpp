#include <opencv2/opencv.hpp>
using namespace cv;

#define eps 1e-3

class myRange {
 public:
  float lower_bound;
  float upper_bound;
  float step;
  std::vector<float> values;

  myRange() 
      : lower_bound(0), upper_bound(1), step(0.1) {
    update();
  }

  myRange(float l, float u, float s)
      : lower_bound(l), upper_bound(u), step(s) {
    update();
  }

  void update() {
    for (float value = lower_bound; value < upper_bound; value += step)
      values.push_back(value);
  }
};
 
int main(int argc, char* argv[]) {
  Ptr<myRange> rng = makePtr<myRange>();
  std::cout << rng << ":" << static_cast<void*>(nullptr) << std::endl;
  std::cout << (rng == nullptr) << std::endl;
  if (rng != nullptr) {
    for (auto &value : rng->values) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }
  std::cin.get();
  return 0;
}