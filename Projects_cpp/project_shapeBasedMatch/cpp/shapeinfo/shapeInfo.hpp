#ifndef SHAPEINFO_PRODUCER
#define SHAPEINFO_PRODUCER

#include "precomp.hpp"

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
    Info() : angle(0), scale(1) {}
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
  cv::Mat src_of(Info info = shapeInfo_producer::Info());

  /// @brief 按日志输出掩图
  /// @param info 读取日志结构体
  /// @return 返回经过旋转和缩放后的掩图矩阵
  cv::Mat mask_of(Info info = shapeInfo_producer::Info());

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

#endif // SHAPEINFO_PRODUCER