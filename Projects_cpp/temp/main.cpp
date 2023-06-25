#include <iostream>
#include <ranges>
#include <algorithm>

#define eps 1e-3

namespace rng = std::ranges;

struct Range {
  float lower_bound;
  float upper_bound;
  float step;
  std::vector<float> values;

  Range(float l, float u, float s)
      : lower_bound(l), upper_bound(u), step(fmax(s, eps)) {
    for (float value = lower_bound; value < upper_bound; value += step)
      values.push_back(value);
  }
};
 
int main(int argc, char* argv[]) {
  Range rg(0.85, 2.01, 0.25);
  for (auto value : rg.values)
    std::cout << value << std::endl;
  return 0;
}