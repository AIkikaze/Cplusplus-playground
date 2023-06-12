#ifndef LINE2D_UP_HPP
#define LINE2D_UP_HPP

#include "MIPP/mipp.h"
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace line2Dup {

#define QUANTIZE_BASE 16

#if QUANTIZE_BASE == 16
#define QUANTIZE_TYPE CV_16U
typedef ushort quantize_type;
#elif QUNATIZE_BASE == 8
#define QUANTIZE_TYPE CV_8U
#endif

struct Feature {
  int x;
  int y;
  int label;
  float alpha;

  Feature() : x(0), y(0), label(0), alpha(0.0f) {}
  Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

struct Template {
  int width;
  int height;
  int _x_;
  int _y_;
  int pyramid_level;
  std::vector<Feature> features;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

  cv::Rect cropTemplate(std::vector<Template> &templs);
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

    Candidate(int _x, int _y, int _label, float _score)
        : f(_x, _y, _label), score(_score) {}

    bool operator<(const Candidate &rhs) const { return score > rhs.score; }
  };

  static void selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                      std::vector<Feature> &features,
                                      size_t num_features, float distance);
};

class Modality {
public:
  virtual ~Modality() {}

  cv::Ptr<QuantizedPyramid> process(const cv::Mat &src,
                                    const cv::Mat &mask = cv::Mat()) {
    return processTempl(src, mask);
  }

  virtual cv::String name() const = 0;

  virtual void read(const cv::FileNode &fn);
  virtual void write(cv::FileStorage &fs) const;

  static cv::Ptr<Modality> create(const cv::String &modality_type);

  static cv::Ptr<Modality> create(const cv::FileNode &fn);

protected:
  virtual cv::Ptr<QuantizedPyramid> processTempl(const cv::Mat &src,
                                                 const cv::Mat &mask) const = 0;
};

class ColorGradient : public Modality {
public:
  ColorGradient();

  ColorGradient(float weak_threshold, float strong_threshold,
                size_t num_features);

  static cv::Ptr<ColorGradient>
  create(float weak_threshold, float strong_threshold, size_t num_features);

  virtual cv::String name() const override;

  virtual void read(const cv::FileNode &fn) override;
  virtual void write(cv::FileStorage &fs) const override;

private:
  float weak_threshold;
  float strong_threshold;
  size_t num_features;

protected:
  virtual cv::Ptr<QuantizedPyramid> processTempl(const cv::Mat &src, const cv::Mat &mask) const override;
};


/// Debug
void colormap(const cv::Mat &quantized, cv::Mat &dst);

void drawFeatures(cv::InputOutputArray img, const std::vector<Template> &templates, const cv::Point2i &_p_, int size = 10);

class ColorGradientPyramid : public QuantizedPyramid {
public:
  ColorGradientPyramid(const cv::Mat &_src, const cv::Mat &_mask, float _weak_threshold,
                       float _strong_threshold, size_t _num_features);

  virtual void quantize(cv::Mat &dst) const override;

  virtual bool extractTemplate(Template &templ) const override;

  virtual void pyrDown() override;

protected:
  void update();

  int pyramid_level;
  cv::Mat src;
  cv::Mat mask;

  cv::Mat magnitude;
  cv::Mat angle;

  float weak_threshold;
  float strong_threshold;
  size_t num_features;
};
} // namespace line2Dup

#endif // LINE2D_UP_HPP