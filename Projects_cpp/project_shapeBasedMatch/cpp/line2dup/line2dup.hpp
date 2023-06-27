#ifndef LINE2D_UP_HPP
#define LINE2D_UP_HPP

#include "precomp.hpp"

class Timer {
 public:
  Timer() : start_(std::chrono::high_resolution_clock::now()), time_(0) {}

  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  void stop() {
    auto end = std::chrono::high_resolution_clock::now();
    time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
                 .count();
    start_ = {};
  }

  double time() {
    stop();
    double ret = static_cast<double>(time_) / 1e9;  // 转换为秒
    time_ = 0;
    return ret;
  }

  void out(const std::string &message = "") {
    double t = time();
    printf("%s\nelapsed time: %fs\n", message.c_str(), t);
    start();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  int64_t time_;
};

class TimeCounter {
 public:
  TimeCounter(cv::String _name) : name(_name), count(0) {}

  void begin() { timer.start(); }
  void countOnce() { count++, time += timer.time(); }
  void out() {
    printf(" ***** %s ***** \n", name.c_str());
    printf(" count -> %7d \n", count);
    printf(" tolal time -> %7fs \n", !count ? 0 : time);
    printf(" average time -> %7fs \n", !count ? 0 : time / count);
  }

 private:
  Timer timer;
  cv::String name;

  int count;
  double time;
};

namespace line2Dup {

/// Feature -> Gradient -> Candidate

/// @brief Point features used for template matching
struct Feature {
  int x;
  int y;
  /// @brief orientation of gradient vector labeled by a discrete number
  /// when mode is 16 bit, the mapping from the angle to the label is
  /// [0, 360) -> [0, 15]; when mode is 8 bit, the mapping would be
  /// [0, 360) -> [0, 7]. The define named by "angle2label" implements
  /// this mapping process
  int label;

  Feature() : x(-1), y(-1), label(0) {}
  Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}
};

/// @brief from the angle of the orientation of gradient vector to the label
/// we used in Feature struct.
#define angle2label(x) (static_cast<int>(x * 32.0f / 360.0f + 0.5f) & 15)

/// @brief adding the extra variable named by "angle". This is for rescaling
/// the featrues when template is resized or rotated.
struct Gradient : Feature {
  float angle;
  Gradient() : Feature(), angle(0) {}
  Gradient(int _x, int _y, float _angle)
      : Feature(_x, _y, angle2label(_angle)), angle(_angle) {}
  Gradient &operator=(const Gradient &rhs) {
    if (this != &rhs) {
      x = rhs.x;
      y = rhs.y;
      label = rhs.label;
      angle = rhs.angle;
    }
    return *this;
  }
};

/// @brief Gradient struct which need to be filtered by the length of
/// the gradient -> float score
struct Candidate : Gradient {
  float score;
  Candidate(int _x, int _y, float _angle, float _score)
      : Gradient(_x, _y, _angle), score(_score) {}

  // need to be sorted by score from largest to smallest
  bool operator<(const Candidate &rhs) const { return score > rhs.score; }
};

/// ShapeTemplate

/// @brief this is a forward declaration of LinearMemory
class LinearMemory;

/// @brief Template struct containing necessary elements for matching
/// calculation and display itself in a background image
struct ShapeTemplate {
  cv::RotatedRect box;             // the RotatedRect contains all featrues
  std::vector<Gradient> features;  // the list of featrues
  int pyramid_level;               // the level in a pyramid
  float scale;                     // the rescaling factor = 1.0f by default
  float angle;                     // the rotation factor  = 0.0f by default

  ShapeTemplate(int _pyramid_level, float _scale, float _angle)
      : pyramid_level(_pyramid_level),
        scale(_scale),
        angle(fmod(_angle + 360.0f, 360.0f)) {}

  /// @brief craete a new template which is rotated by float new_angle and
  /// rescaled by flaot new_scale, then manage its lifecycle using cv::Ptr
  /// @param new_scale[in] rewrite the scale of the ShapeTemplate
  /// @param new_angle[in] rewrite the angle of the ShapeTemplate
  cv::Ptr<ShapeTemplate> relocate(float new_scale, float new_angle);

  /// @brief show all features and draw the box centered by cv::Point
  /// new_center in a given image as background
  /// @param background[in/out] background image which will be modified in
  /// this function
  /// @param new_center[in] the point to determine the location of the
  /// template
  void show_in(cv::Mat &background, cv::Point new_center);

  /// @brief show all features which is colored by its match score and
  /// draw the box centered by cv::Point new_center in a given image as
  /// background
  /// @param background[in/out] background image which will be modified in
  /// this function
  /// @param score_maps[in] linearied response map used to find the
  /// match socre of each feature point
  /// @param new_center[in] the point to determine the location of the
  /// template
  void show_in(cv::Mat &background, std::vector<LinearMemory> &score_maps,
               cv::Point new_center);
};

/// Range -> Search

/// @brief Range of rescaling factor or rotation factor
struct Range {
  float lower_bound;
  float upper_bound;
  float step;
  std::vector<float> values;

  Range()
      : lower_bound(0), upper_bound(0), step(line2d_eps) {}

  Range(float l, float u, float s)
      : lower_bound(l), upper_bound(u), step(fmax(s, line2d_eps)) {
    update();
  }

  Range(const float (&lus)[3])
      : lower_bound(lus[0]), upper_bound(lus[1]), step(lus[2]) {
    update();
  }

  void setStep(float new_step) {
    step = new_step;
    update();
  }

  void update() {
    values.clear();
    for (float value = lower_bound; value < upper_bound; value += step) {
      values.push_back(value);
    }
  }
};

/// @brief Search in scale range and angle range
struct Search {
  Range scale;
  Range angle;
  Search() : scale(1, 1, line2d_eps), angle(0, 0, line2d_eps) {}
  Search(Range _scale, Range _angle) : scale(_scale), angle(_angle) {}
};

/// Template Search

/// @brief Using Search struct and ShapeTemplate struct to create a
/// list of templates rotated and rescaled by coordinate
/// (scale, angle). You can use (scale, angle) to search a sublist
/// of templates which are adjacent to the coordinate you give.
class TemplateSearch {
 public:
  TemplateSearch() : rows(0), cols(0) {}

  cv::Ptr<ShapeTemplate> &operator[](int id) { return templates[id]; }

  /// @brief return the total size of the list of templates in class
  /// @return an int variable -> templates.size()
  int size() { return templates.size(); }

  /// @brief use (scale, angle) to search a sublist of templates
  /// which are adjacent to the coordinate you give.
  /// @param scale[in] the float scale represents the rescaling factor
  /// @param angle[in] the float angle represents the rotation factor
  /// @return a vector of Ptr<ShapeTemplate> represents the targeted
  /// sublist of templates
  std::vector<cv::Ptr<ShapeTemplate>> searchInRegion(float scale, float angle);

  /// @brief use Search struct and ShapeTemplate struct to create this
  /// class, in another word, to initialize the list of templates rotated
  /// and rescaled by coordinates in the Search area
  /// @param search[in] you need to provide the Search area
  /// @param base[in] you need to provide an original template as the
  /// base. "Original" means its scale would better be 1.0f, and its angle
  /// would better be 0.0f
  void build(const Search &search, ShapeTemplate &base);

 private:
  int rows;
  int cols;
  Search region;
  std::vector<cv::Ptr<ShapeTemplate>> templates;
};

/// ColorGradientPyramid

/// @brief this is a class possessing 2 basic function :
/// 1. calculate the quantized mat of the phase mat of a source image
/// and export it using "quantize" function
/// 2. exact Template from the quantized mat to a ShapeTemplate in
/// "extractTemplate" function
class ColorGradientPyramid {
 public:
  float magnitude_threshold;
  int count_kernel_size;
  size_t num_features;

  ColorGradientPyramid()
      : magnitude_threshold(50.0f), count_kernel_size(3), num_features(100) {}

  /// @brief call constructor to create a new instance of ColorGradientPyramid
  /// @return Ptr<ColorGradientPyramid>
  void process(const cv::Mat &_src,
               const cv::Mat &_mask = cv::Mat()) {
    src = _src;
    mask = _mask;
    update();
  }

  /// @brief copy quantized mat to dst
  /// @param dst[out] empty mat is OK
  void quantize(cv::Mat &dst) const {
    dst = cv::Mat::zeros(quantized_angle.size(), quantized_angle.type());
    quantized_angle.copyTo(dst, mask);
  }

  /// @brief exact Template from the quantized mat
  /// @param templ[out] empty ShapeTemplate is ok
  /// @return a bool variable presents whether Template is well-defined --
  /// we get enough number of features to reach the num_feature
  bool extractTemplate(ShapeTemplate &templ) const;

  /// @brief resize the source image and mask
  void pyrDown();

  /// @brief to get the source Image as background
  cv::Mat background() { return src.clone(); }

 private:
  // ColorGradientPyramid(const cv::Mat &_src, const cv::Mat &_mask,
  //                      float _magnitude_threshold = 50.0f,
  //                      int _count_kernel_size = 3, size_t _num_features = 100);

  /// @brief recalculate the quantized mat of the phase mat of the
  /// source image
  void update();


  int pyramid_level = 0;

  cv::Mat src;
  cv::Mat mask;

  cv::Mat magnitude;
  cv::Mat angle;
  cv::Mat quantized_angle;
};

/// Match and Detector

/// @brief Match Point struct
struct Match {
  int x;
  int y;
  float similarity;
  cv::Ptr<ShapeTemplate> templ;

  Match(int _x, int _y, float _similarity, cv::Ptr<ShapeTemplate> _template)
      : x(_x), y(_y), similarity(_similarity), templ(_template) {}

  bool operator<(const Match &rhs) const { return similarity < rhs.similarity; }

  bool operator==(const Match &rhs) const {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity &&
           templ == rhs.templ;
  }
};

/// @brief The data structure obtained by repartitioning the matrix into 4x4
/// blocks
class LinearMemory {
 public:
  int block_size;
  int rows;
  int cols;

  LinearMemory() 
      : block_size(4), rows(0), cols(0) {}
  
  LinearMemory(int _block_size) 
      : block_size(_block_size), rows(0), cols(0) {
    memories.resize(block_size * block_size);
  }

  size_t linear_size() { return memories[0].size(); }

  size_t size() { return memories.size() * memories[0].size(); }

  void create(size_t size, short value = 0) {
    for (int i = 0; i < (int)memories.size(); i++)
      memories[i] = std::vector<short>(size, value);
  }

  /// @brief to access memories[i][j] -> linearized Mat S_{orientaion}(x)
  /// @param i -> order in TxT block
  /// @param j -> index in linear vector
  /// @return &memories[i][j]
  inline short &at(const int &i, const int &j) {
    static short default_value = 0;  // 静态的默认值
    if (i < 0 || i >= (int)memories.size()) return default_value;
    if (j < 0 || j >= (int)memories[0].size()) return default_value;
    return memories[i][j];
  }

  /// @brief to access memories(y, x) using Mat coordinate (y, x)
  /// @param y[in] the y coordinate in Rows x Cols Mat
  /// @param x[in] the x coordinate in Rows x Cols Mat
  /// @return &memories(y, x)
  inline short &linear_at(const int &y, const int &x) {
    int i = (y % block_size) * block_size + (x % block_size);
    int j = (y / block_size) * cols + (x / block_size);
    return at(i, j);
  }

  /// @brief to create LinearMemory from a Mat
  /// @param src[in] the source Mat
  void linearize(const cv::Mat &src);

  /// @brief copy LinearMemory to a Mat
  /// @param dst[out] the copied Mat would be CV_16U and resized by
  /// (rows * block_size) x (cols * block_size)
  void unlinearize(cv::Mat &dst);

 private:
  std::vector<std::vector<short>> memories;
};

class Detector {
 public:
  Detector(int _pyramid_level, int _block_size,
           std::vector<int> _spread_kernels,
           cv::Ptr<ColorGradientPyramid> _modality)
      : name("default"),
        pyramid_level(_pyramid_level),
        block_size(_block_size),
        spread_kernels(_spread_kernels),
        modality(_modality) {
    CV_Assert(pyramid_level > 0);
    CV_Assert(block_size > 2);
    CV_Assert((int)spread_kernels.size() >= pyramid_level);
    CV_Assert(modality != nullptr);
  }

  Detector(cv::String _name, int _pyramid_level, int _block_size,
           std::vector<int> _spread_kernels,
           cv::Ptr<ColorGradientPyramid> _modality)
      : name(_name),
        pyramid_level(_pyramid_level),
        block_size(_block_size),
        spread_kernels(_spread_kernels),
        modality(_modality) {
    CV_Assert(pyramid_level > 0);
    CV_Assert(block_size > 2);
    CV_Assert((int)spread_kernels.size() > pyramid_level);
    CV_Assert(modality != nullptr);
  }

  void setSource(cv::Mat &src, cv::Mat mask = cv::Mat());

  void setTemplate(cv::Mat &object, cv::Mat object_mask = cv::Mat());

  void setTemplate(cv::Mat &object, cv::Mat object_mask,
                   const float (&scale_range)[3],
                   const float (&angle_range)[3]);

  void match(float score_threshold);

  void detectBestMatch(std::vector<cv::Vec6f> &points,
                       std::vector<cv::RotatedRect> &boxs);

  void draw(cv::Mat background);

 private:
  Detector() {}

  cv::String name;
  int pyramid_level;
  int block_size;
  std::vector<int> spread_kernels;
  cv::Ptr<ColorGradientPyramid> modality;

  std::vector<TemplateSearch> templates_pyramid;
  std::vector<LinearMemory> memories_pyramid;
  std::vector<Match> matches;
};

}  // namespace line2Dup

#endif  // LINE2D_UP_HPP

