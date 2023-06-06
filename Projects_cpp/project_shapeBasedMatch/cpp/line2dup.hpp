#ifndef LINE2D_UP_HPP
#define LINE2D_UP_HPP

#include "MIPP/mipp.h"
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/rgbd/linemod.hpp>

namespace line2Dup {

struct Feature {
  int x;
  int y;
  int label;
  float alpha;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

  Feature() : x(0), y(0), label(0) {}
  Feature(int x, int y, int labelk);
};

inline Feature::Feature(int x, int y, int label) : x(x), y(y), label(label) {}

struct Template {
  int width;
  int height;
  int _x_;
  int _y_;
  int pyramid_level;
  std::vector<Feature> featrues;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

class QuantizedPyramid {
public:
  virtual ~QuantizedPyramid() {}

  virtual void quantize(cv::Mat &dst) const = 0;

  virtual bool extractTemplate(Template &templ) const = 0;

  virtual void pyrDown() = 0;

protected:
  struct Candidate {
    Feature f;
    float score;

    Candidate(int x, int y, int label, float score);

    bool operator<(const Candidate &rhs) const { return score > rhs.score; }
  };

  static void selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                      std::vector<Feature> &features,
                                      size_t num_features, float distance);
};

inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label,
                                              float score)
    : f(x, y, label), score(score) {
}

class Modality {
public:
  virtual ~Modality() {}

  virtual cv::Ptr<QuantizedPyramid>
  process(const cv::Mat &src, const cv::Mat &mask = cv::Mat()) const = 0;

  virtual cv::String name() const = 0;

  virtual void read(const cv::FileNode &fn);
  virtual void write(cv::FileStorage &fs) const;

  static cv::Ptr<Modality> create(const cv::String &modality_type);

  static cv::Ptr<Modality> create(const cv::FileNode &fn);

  protected:
};

} // namespace line2Dup

#endif // LINE2D_UP_HPP