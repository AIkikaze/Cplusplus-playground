#ifndef OPENCV_LINE_2D_HPP
#define OPENCV_LINE_2D_HPP

#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define line2d_eps 1e-7f

/// @brief 图像金字塔
class ImagePyramid {
 public:
  ImagePyramid();

  ImagePyramid(const cv::Mat &src, int level_size);

  const cv::Mat &operator[](int index) const;

 private:
  int levels;                    // 图像金字塔层级
  std::vector<cv::Mat> pyramid;  // 图像 vector 序列: 最底层为裁剪后的原始图像,
                                 // 最高层为缩放 1<<levels 倍的图像

  /// @brief 读取图像以初始化图像金字塔
  /// @param src 输入图像矩阵
  /// @param levels 金字塔最高层数evels
  void buildPyramid(const cv::Mat &src, int levels = 0);
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
  shapeInfo_producer(const cv::Mat &input_src, cv::Mat input_mask,
                     bool padding);

  /// @brief
  /// @param sip
  /// @param path
  static void save_config(const shapeInfo_producer &sip, std::string path);

  /// @brief
  /// @param input_src
  /// @param input_mask
  /// @param padding
  /// @param path
  /// @return
  static cv::Ptr<shapeInfo_producer> load_config(const cv::Mat &input_src,
                                                 cv::Mat input_mask,
                                                 bool padding,
                                                 std::string path);

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

/// @todo 完成模板 template 类
class Template {
 public:
  int num_prograds;        // 特征方向个数
  float lowest_grad_norm;  // 最小梯度阈值 取值范围 [0, 1.0)

  struct Features {
    cv::Point p_xy;    // 特征点坐标
    ushort angle_bit;  // 特征方向的 16 位二进制表示
    float grad_norm;   // 梯度模长(已归一化)
    Features(cv::Point xy, ushort angle, float grad)
        : p_xy(xy), angle_bit(angle), grad_norm(grad) {}
    bool operator<(const ProGrads &rhs) const {  // 按梯度模长进行排序
      return grad_norm < rhs.grad_norm;
    }
  };

  Template();

  static ushort angle2bit(const float &angle);

  static void getOriMat(const Template &inputTemplate, cv::Mat &edges,
                        cv::Mat &angles);

  static void getFeatures(Template &inputTemplate, cv::Mat &edges,
                        cv::Mat &angles, int kernel_size);

  static void scatterFeatures(const Template &inputTemplate);

  static void exactFeatures(const Template &inputTemplate);

  static void cropTemplate(Template &inputTemplate);

  vector<Template::Features> &pg_ptr();

 private:
  std::vector<Template::Features> prograds;
  cv::Mat src;
  cv::Mat mask;
};

/// @todo 完成检测算子类
class Detector {
 public:
  Detector() {}
  void addTemplate(const cv::Mat &src, const cv::Mat &mask,
                   int num_features = 0) {}

 private:
};

#endif  // OPENCV_LINE_2D_HPP

void init_costable() {
  int maxValue = std::numeric_limits<ushort>::max();
  cos_table = std::vector<float>(16 * maxValue);
  float maxCos;
  for (int i = 1; i <= maxValue; i++) {
    for (int j = 0; j < 16; j++) {
      maxCos = 0.0f;
      for (int k = 0; k < 16; k++) {
        if ((ushort(i)) & (1 << k))
          maxCos = maxCos < abs(cos(bit2angle(1 << k) - bit2angle(1 << j)))
                       ? abs(cos(bit2angle(1 << k) - bit2angle(1 << j)))
                       : maxCos;
      }
      cos_table[i * 16 + j] = maxCos;
    }
  }
}

inline static float bit2angle(const ushort &angle_bit) {
  float init_angle = 180.0f / 32.0f;
  for (int i = 0; i < 16; i++) {
    if (angle_bit & (1 << i)) {
      return init_angle + (180.0f / 16.0f) * i;
    }
  }
  return 0.0f;  // angle_bit == 0
}