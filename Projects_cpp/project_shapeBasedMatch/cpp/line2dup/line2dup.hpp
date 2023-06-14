#ifndef LINE2D_UP_HPP
#define LINE2D_UP_HPP

#include "precomp.hpp"

namespace line2Dup {

#define line2d_eps 1e-7f
#define _degree_(x) ((x)*CV_PI) / 180.0

#define QUANTIZE_BASE 16

#if QUANTIZE_BASE == 16
#define QUANTIZE_TYPE CV_16U
typedef ushort quantize_type;
#elif QUNATIZE_BASE == 8
#define QUANTIZE_TYPE CV_8U
#endif

// Feature -> Gradient -> Candidate

struct Feature {
  int x;
  int y;
  int label;
  Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

#define angle2label(x) (static_cast<int>(x * 32.0 / 360.0) & 15)

struct Gradient : Feature {
  float angle;
  Gradient(int _x, int _y, float _angle)
      : Feature(_x, _y, angle2label(_angle)), angle(_angle) {}
};

/// @brief 待筛选的特征结构体
struct Candidate : Feature {
  float score;
  Candidate(int _x, int _y, int _label, float _score)
      : Feature(_x, _y, _label), score(_score) {}

  bool operator<(const Candidate &rhs) const { return score > rhs.score; }
};

// Template -> ShapeTemplate

struct Template {
  cv::RotatedRect box;
  int pyramid_level;
  std::vector<Feature> features;

  // 数据存储与读取
  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

class ShapeTemplate {
public:
  cv::RotatedRect box;
  int pyramid_level;
  std::vector<Gradient> features;
  float scale;
  float angle;

  ShapeTemplate() : scale(1.0f), angle(0.0f), lazy(true) {}
  ShapeTemplate(float _alpha, float _theta, bool _lazy)
      : scale(_alpha), angle(_theta), lazy(_lazy) {}

  // 加载旋转缩放
  static void relocate(ShapeTemplate &templ);

  // 懒加载
  inline void process() {
    if (lazy)
      relocate(*this);
  }

  // 类型转化
  operator Template() const;

private:
  bool lazy; // 1 -> 未进行缩放旋转; 0 -> 已进行缩放旋转
};

/// ColorGradientPyramid

/// @brief 1. 计算梯度方向的量化矩阵 2. 将提取模型
class ColorGradientPyramid {
public:
  ColorGradientPyramid(const cv::Mat &_src, 
                       const cv::Mat &_mask,
                       float _magnitude_threshold = 80.0f, 
                       int count_kernel_size = 5,
                       size_t _num_features = 100);

  cv::Ptr<ColorGradientPyramid> process(const cv::Mat src,
                                        const cv::Mat &mask = cv::Mat()) const {
    return cv::makePtr<ColorGradientPyramid>(src, mask, magnitude_threshold,
                                             count_kernel_size, num_features);
  }

  cv::Mat quantized_angle() const {
    cv::Mat quantized_angle;
    angle.copyTo(quantized_angle, mask);
    return quantized_angle;
  }

  bool extractTemplate(Template &templ) const;

  void pyrDown();

private:
  inline void update();

  int pyramid_level;

  cv::Mat src;
  cv::Mat mask;

  cv::Mat magnitude;
  cv::Mat angle;

  float magnitude_threshold;
  int count_kernel_size;
  size_t num_features;
};

/// Search

struct Range {
  float lower_bound;
  float upper_bound;
  float step;
  Range(float l, float u, float s) : lower_bound(l), upper_bound(u), step(s) {}
  Range(float range_params[3])
      : lower_bound(range_params[0]), upper_bound(range_params[1]),
        step(range_params[2]) {}
};

struct Search {
  Range scale;
  Range angle;
  Search(Range _scale, Range _angle) : scale(_scale), angle(_angle) {}
};

/// Match and Detector

struct Match {
  int x;
  int y;
  float similarity;
  cv::String class_id;
  int template_id;
  Match(int _x, int _y, float _similarity, const cv::String &_class_id,
        int _template_id)
      : x(_x), y(_y), similarity(_similarity), class_id(_class_id),
        template_id(_template_id) {}

  bool operator<(const Match &rhs) const {
    if (similarity != rhs.similarity)
      return similarity < rhs.similarity;
    else
      return template_id < rhs.template_id;
  }

  bool operator==(const Match &rhs) const {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity &&
           class_id == rhs.class_id;
  }
};

class LinearMemory {
public:
  LinearMemory(int _block_size, cv::Size size)
    : block_size(_block_size), rows(size.height), cols(size.width) { 
    memories = std::vector<std::vector<ushort>>(block_size * block_size, std::vector<ushort>(cols * rows, 0));
  }
  
  void linearize(cv::Mat &src);

  void unlinearize(cv::Mat &dst);

  void read(cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

private:
  std::vector<std::vector<ushort>> memories;
  int block_size;
  int rows;
  int cols;
};

class Detector {
public:
  // src & mask -> response map
  void setSource(cv::Mat &src, cv::Mat mask = cv::Mat());

  // src & mask -> shapetemplate
  void setTemplate(cv::Mat &object, cv::Mat object_mask = cv::Mat());

  void addSource(cv::Mat &src, cv::Mat mask = cv::Mat(), const cv::String &src_name = "default");

  void addTemplate(cv::Mat &object, cv::Mat object_mask = cv::Mat(), const cv::String &templ_name = "default");

  void addSearch(Range &scale, Range &angle);

  void match(cv::Mat &src, cv::Mat &object, 
             float score_threshold,
             cv::Mat src_mask = cv::Mat(), 
             cv::Mat object_mask = cv::Mat());

  void match(cv::String &match_name, cv::String &search_name);

private:
  cv::Ptr<ColorGradientPyramid> modality;
  std::map<cv::String, std::vector<ShapeTemplate> > templates_map;
  std::map<cv::String, std::vector<LinearMemory> > memories_map;
  std::map<cv::String, std::vector<Search> > searches_map;

  int pyramid_level;
  std::list<int> spread_kernel_size;
  std::list<int> gaussian_kernel_size;
};
} // namespace line2Dup

#endif // LINE2D_UP_HPP