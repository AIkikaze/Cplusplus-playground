#ifndef OPENCV_LINE_2D_HPP
#define OPENCV_LINE_2D_HPP

#include <array>
#include <bitset>
#include <chrono>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

class Timer {
 public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
  }
  void out(std::string message = "") {
    double t = elapsed();
    std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
    reset();
  }

 private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;
};

#define line2d_eps 1e-7f
#define _degree_(x) ((x)*CV_PI) / 180.0

namespace line2d {

/// @brief 图像金字塔
class ImagePyramid {
 public:
  ImagePyramid();

  ImagePyramid(const cv::Mat &src, int py_level);

  cv::Mat &operator[](int index);

  /// @brief 读取图像以初始化图像金字塔
  /// @param src 输入图像矩阵
  /// @param pyramid_level 金字塔最高层数evels
  void buildPyramid(const cv::Mat &src, int pyramid_level = 0);

 private:
  int pyramid_level;  // 图像金字塔层级
  std::vector<cv::Mat> pyramid;  // 图像 vector 序列: 最底层为裁剪后的原始图像,
                                 // 最高层为缩放 1<<pyramid_level 倍的图像
};

/// @brief 形状日志生成器
class shapeInfo_producer {
 public:
  std::array<float, 2> angle_range;  // 角度范围
  std::array<float, 2> scale_range;  // 缩放系数范围
  float angle_step;                  // 角度步长
  float scale_step;                  // 缩放系数步长
  float eps;                         // 搜索精度

  /// @brief 日志结构体, 含旋转角和缩放系数两个参数
  struct Info {
    float angle;
    float scale;
    Info(float input_angle, float input_scale)
        : angle(input_angle), scale(input_scale) {}
  };

  /// @brief 初始化形状日志生成器的公有成员
  shapeInfo_producer();

  /// @brief 用构造函数读入模板图像 src 和掩图 mask, 布尔变量 padding 默认为
  /// true 表示当前读入的图像已经扩充好的 0 边界,
  /// 否则在构造函数中使用默认方法以扩充图像边界。
  /// @param input_src 待读入的模板图像 src
  /// @param input_mask 待读入的掩图 mask, 默认为与 src 等大小的全 255 像素矩阵
  /// @param padding 边界扩张选项，默认为 false
  shapeInfo_producer(const cv::Mat &input_src, cv::Mat input_mask = cv::Mat(),
                     bool padding = false);

  /// @brief
  /// @param sip
  /// @param path
  static void save_config(const shapeInfo_producer &sip,
                          std::string path = "../data/sip_config.yaml");

  /// @brief
  /// @param input_src
  /// @param input_mask
  /// @param padding
  /// @param path
  /// @return
  static cv::Ptr<shapeInfo_producer> load_config(
      const cv::Mat &input_src, cv::Mat input_mask = cv::Mat(),
      bool padding = false, std::string path = "../data/sip_config.yaml");

  /// @brief 按当前设置读入形状日志
  void produce_infos();

  /// @brief 以旋转角 angle 和缩放系数 scale 对输入图像 src 作仿射变换,
  /// 返回经过仿射变换后的像素矩阵
  /// @param src 输入图像
  /// @param angle 旋转角
  /// @param scale 缩放系数
  /// @return 经过仿射变换后的矩阵
  static cv::Mat affineTrans(const cv::Mat &src, float angle, float scale);

  /// @brief 按日志输出模板图像
  /// @param info 读取日志结构体
  /// @return 返回经过旋转和缩放后的模板图像矩阵
  cv::Mat src_of(const Info &info);

  /// @brief 按日志输出掩图
  /// @param info 读取日志结构体
  /// @return 返回经过旋转和缩放后的掩图矩阵
  cv::Mat mask_of(const Info &info);

  /// @brief 返回日志列表 Infos 的常量指针, 不可修改列表成员
  /// @return 类中的私有日志列表 Infos
  const std::vector<shapeInfo_producer::Info> &Infos_constptr() const;

 private:
  std::vector<shapeInfo_producer::Info> Infos;  // 日志列表
  cv::Mat src;                                  // 模板图像矩阵
  cv::Mat mask;                                 // 模板对应掩图矩阵

  /// @brief
  /// @return
  std::vector<shapeInfo_producer::Info> &Infos_ptr() { return Infos; }
};

inline int angle2ori(const float &angle);
inline float ori2angle(const int &orientation);

class Template {
 public:
  struct Features {
    cv::Point p_xy;   // 特征点坐标
    float angle;      // 特征方向角度 0~360
    float grad_norm;  // 梯度模长(已归一化)
    int orientation;  // 特征方向角量化后的标签 0~15
    Features(cv::Point xy, float angle, float grad)
        : p_xy(xy), angle(angle), grad_norm(grad) {}
    // bool operator<(const Features &rhs) const {  // 按梯度模长进行排序
    //   return grad_norm < rhs.grad_norm;
    // }
  };

  struct TemplateParams {
    int num_features;
    float grad_norm;
    bool template_created;
    int nms_kernel_size;
    float scatter_distance;
    bool isDefault;

    TemplateParams() {
      num_features = 200;
      grad_norm = 0.2f;
      template_created = false;
      nms_kernel_size = 7;
      scatter_distance = 10.0f;
      isDefault = true;
    }

    // 重载赋值运算符
    TemplateParams &operator=(const TemplateParams &other) {
      if (this == &other) {
        return *this;
      }
      num_features = other.num_features;
      grad_norm = other.grad_norm;
      template_created = other.template_created;
      nms_kernel_size = other.nms_kernel_size;
      scatter_distance = other.scatter_distance;
      isDefault = false;
      return *this;
    }
  };

  Template();

  Template(TemplateParams params, bool isDefault = false);

  static void getOriMat(const cv::Mat &src, cv::Mat &edges, cv::Mat &angles);

  static void createTemplate(const cv::Mat &src, Template &tp,
                             int kernel_size = 3,
                             float scatter_distance = 6.0f);

  static cv::Ptr<Template> createPtr_from(
      const cv::Mat &src, TemplateParams params = TemplateParams());

  void create_from(const cv::Mat &src);

  void selectFeatures_from(const cv::Mat &_edges, const cv::Mat &_angles,
                           int nms_kernel_size = 3);

  void scatter(float upper_distance = 6.0f);

  std::vector<Features> relocate_by(shapeInfo_producer::Info info);

  bool iscreated() { return template_created; }

  const std::vector<Features> &pg_ptr() const { return prograds; };

  void show_in(const cv::Mat &background, cv::Point center);

 private:
  TemplateParams defaultParams;    // 默认参数列表
  int nms_kernel_size;             // 非极大值抑制的窗口大小
  float scatter_distance;          // 离散化特征点之间的最小距离
  float grad_norm;                 // 最小梯度阈值 取值范围 [0, 1.0)
  size_t num_features;             // 特征方向个数
  bool template_created;           // 记录模板创建状态
  std::vector<Features> prograds;  // 特征方向序列
};

class Detector {
 public:
  struct MatchPoint {
    cv::Point p_xy;
    float similarity;
    MatchPoint(cv::Point p, float similar_score)
        : p_xy(p), similarity(similar_score) {}
    bool operator<(const MatchPoint &rhs) const {
      return similarity > rhs.similarity;
    }
  };

  class LinearMemories {
   private:
    std::vector<std::vector<float>> memories;  // 线性存储器

   public:
    int rows;
    int cols;

    int linear_size() { return memories[0].size(); }

    void create(size_t x, size_t y, float value = 0.0f) {
      memories = std::vector<std::vector<float>>(x, std::vector<float>(y, value));
    }

    /// @brief memories[i][j] -> linearized Mat S_{orientaion}(c)
    /// @param i -> order in TxT kernel
    /// @param j -> index in linear vector
    /// @return &memories[i][j]
    float &at(size_t i, size_t j) {
      static float default_value = 0.0f;  // 静态的默认值
      if (i < 0 || i >= memories.size()) return default_value;
      if (j < 0 || j >= memories[0].size()) return default_value;
      return memories[i][j];
    }
  };

  int pyramid_level;
  ImagePyramid src_pyramid;
  ImagePyramid temp_pyramid;
  std::vector<std::vector<LinearMemories>> memory_pyramid;
  std::vector<cv::Ptr<Template>> temps;
  std::vector<MatchPoint> match_points;
  static std::vector<float> cos_table;

  Detector();

  static void quantize(const cv::Mat &edges, const cv::Mat &angles,
                       cv::Mat &dst, int kernel_size = 3,
                       float grad_norm = line2d_eps);

  static void spread(cv::Mat &ori_bit, cv::Mat &spread_ori,
                     int kernel_size = 3);

  static void computeResponseMaps(cv::Mat &spread_ori,
                                  std::vector<cv::Mat> &response_maps);

  static void computeSimilarityMap(std::vector<LinearMemories> &memories,
                                   Template &temp, LinearMemories &similarity);

  static void localSimilarityMap(std::vector<LinearMemories> &memories,
                                 Template &temp, cv::Mat &similarity_map,
                                 std::vector<cv::Rect> &roi_list);

  static void linearize(std::vector<cv::Mat> &response_maps,
                        std::vector<LinearMemories> &linearized_memories);

  static void unlinearize(LinearMemories &similarity, cv::Mat &similarity_map);

  static void produceRoi(cv::Mat &similarity_map,
                         std::vector<cv::Rect> &roi_list, int lower_score);

  void setSourceImage(const cv::Mat &src, int pyramid_level = 2,
                      cv::Mat mask = cv::Mat());

  void setTempate(const cv::Mat &temp_src, int pyramid_level = 2,
                  Template::TemplateParams params = Template::TemplateParams(),
                  cv::Mat mask = cv::Mat());

  void selectMatchPoints(cv::Mat &similarity_map,
                         std::vector<cv::Rect> &roi_list, int lower_score);

  void match(const cv::Mat &sourceImage, const cv::Mat &templateImage,
             int lower_score = 90,
             Template::TemplateParams params = Template::TemplateParams(),
             cv::Mat mask_src = cv::Mat(), cv::Mat mask_temp = cv::Mat());

 private:
  void init_costable();
};

}  // namespace line2d

#endif  // OPENCV_LINE_2D_HPP