#ifndef LINE2D_UP_HPP
#define LINE2D_UP_HPP

#include "precomp.hpp"

namespace line2Dup {

#define QUANTIZE_BASE 16

#if QUANTIZE_BASE == 16
#define QUANTIZE_TYPE CV_16U
typedef ushort quantize_type;
#elif QUNATIZE_BASE == 8
#define QUANTIZE_TYPE CV_8U
#endif

/// @brief 特征点结构体
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

/// @brief 模型结构体
struct Template {
  int width;                     // 重新裁剪后的宽度
  int height;                    // 重新裁剪后的高度
  int _x_;                       // 定位点横坐标
  int _y_;                       // 定位点纵座标
  int pyramid_level;             // 金字塔层级
  std::vector<Feature> features; // 特征点序列

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

  cv::Rect cropTemplate(std::vector<Template> &templs);
};

/// @brief 量化金字塔
class QuantizedPyramid {
public:
  virtual ~QuantizedPyramid() {}

  /// @brief 从梯度矩阵中得出量化方向
  virtual void quantize(cv::Mat &dst) const = 0;

  /// @brief 从模板中提出特征点序列
  virtual bool extractTemplate(Template &templ) const = 0;

  /// @brief 金字塔下采样
  virtual void pyrDown() = 0;

protected:
  /// @brief 待筛选的特征结构体
  struct Candidate {
    Feature f;
    float score;

    Candidate(int _x, int _y, int _label, float _score)
        : f(_x, _y, _label), score(_score) {}

    bool operator<(const Candidate &rhs) const { return score > rhs.score; }
  };

  /// @brief 离散化特征
  /// @param[in] candidates 待筛选的结构体
  /// @param[out] features 筛选后的结构体
  /// @param[in] num_features 必须达到的特征点个数
  /// @param[in] distance 离散化的点间距
  static void selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                      std::vector<Feature> &features,
                                      size_t num_features, float distance);
};

/// @brief 模型类
class Modality {
public:
  virtual ~Modality() {}

  /// @brief 模型初始化接口，从源图像和遮罩中创建量化金字塔的类对象，并用指针
  /// Ptr 管理
  /// @param[in] src 源图像
  /// @param[in] mask 遮罩
  cv::Ptr<QuantizedPyramid> process(const cv::Mat &src,
                                    const cv::Mat &mask = cv::Mat()) {
    return processTempl(src, mask);
  }

  virtual cv::String name() const = 0;

  virtual void read(const cv::FileNode &fn) = 0;
  virtual void write(cv::FileStorage &fs) const = 0;

  static cv::Ptr<Modality> create(const cv::String &modality_type);

  static cv::Ptr<Modality> create(const cv::FileNode &fn);

protected:
  virtual cv::Ptr<QuantizedPyramid> processTempl(const cv::Mat &src,
                                                 const cv::Mat &mask) const = 0;
};

/// @brief 色彩梯度
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
  virtual cv::Ptr<QuantizedPyramid>
  processTempl(const cv::Mat &src, const cv::Mat &mask) const override;
};

/// Debug
void colormap(const cv::Mat &quantized, cv::Mat &dst);

void drawFeatures(cv::InputOutputArray img,
                  const std::vector<Template> &templates,
                  const cv::Point2i &_p_, int size = 10);

/// @brief 色彩梯度金字塔
class ColorGradientPyramid : public QuantizedPyramid {
public:
  ColorGradientPyramid(const cv::Mat &_src, const cv::Mat &_mask,
                       float _weak_threshold, float _strong_threshold,
                       size_t _num_features);

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

class Detector {
public:
  Detector();

  Detector(const std::vector<cv::Ptr<Modality>> &modalities,
           const std::vector<int> &temp_pyramid);

  void
  match(const std::vector<cv::Mat> &sources, float threshold,
        std::vector<Match> &matches,
        const std::vector<cv::String> &class_ids = std::vector<cv::String>(),
        cv::OutputArrayOfArrays quantized_images = cv::noArray(),
        const std::vector<cv::Mat> &masks = std::vector<cv::Mat>()) const;

  int addTemplate(const std::vector<cv::Mat> &sources,
                  const cv::String &class_id, const cv::Mat &object_mask,
                  cv::Rect *bounding_box = NULL);

  int addSynthicTemplate(const std::vector<Template> &templates,
                         const cv::String &class_id);

  const std::vector<cv::Ptr<Modality>> &getModalities() const {
    return modalities;
  }

  int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

  int pyramidLevels() const { return pyramid_levels; }

  const std::vector<Template> &getTemplates(const cv::String &class_id,
                                            int template_id) const;

  int numTemplates() const;
  int numTemplates(const cv::String &class_id) const;
  int numClasses() const { return static_cast<int>(class_templates.size()); }

  std::vector<cv::String> classIds() const;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

  cv::String readClass(const cv::FileNode &fn,
                       const cv::String &class_id_override = "");
  void writeClass(const cv::String &class_id, cv::FileStorage &fs) const;
  void readClass(const std::vector<cv::String> &class_ids,
                 const cv::String &format = "templates_%s.yml.gz");
  void writeClass(const cv::String &format = "templates_%s.yml.gz") const;

protected:
  std::vector<cv::Ptr<Modality>> modalities; // 待匹配的模型列表
  int pyramid_levels;                        // 金字塔层数
  std::vector<int> T_at_level;               // 各层使用的响应核大小

  typedef std::vector<Template>
      TemplatePyramid;                       // 定义：模板金字塔 -> 用于储存各层各参数的模板 
                                             // tp[i * num_modalities + j] -> 第 i 层中第 j 个模板
  typedef std::map<cv::String, std::vector<TemplatePyramid>>
      TemplateMap;                           // 定义：模板映射 -> 用于查找模板
  TemplateMap class_templates;

  typedef std::vector<cv::Mat> LinearMemories;
  typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid &lm_pyramid,
                  const std::vector<cv::Size> &sizes, float threshold,
                  std::vector<Match> &matches, const cv::String &class_id,
                  const std::vector<TemplatePyramid> &template_pyramids) const;
};

cv::Ptr<Detector> getDefaultLINE();

} // namespace line2Dup

#endif // LINE2D_UP_HPP