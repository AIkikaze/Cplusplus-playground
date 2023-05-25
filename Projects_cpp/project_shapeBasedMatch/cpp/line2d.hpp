#ifndef OPENCV_LINE_2D_HPP
#define OPENCV_LINE_2D_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

/// @brief 图像金字塔
class ImagePyramid {
 public:
  ImagePyramid() {
    levels = 0;
    pyramid = std::vector<cv::Mat>();
  }

  ImagePyramid(const cv::Mat &src, int level_size) {
    levels = level_size;
    buildPyramid(src, levels);
  }

  // 重载 [] 来访问图像金字塔的各层图像
  const cv::Mat &operator[](int index) const {
    CV_Assert(index >= 0 && index < levels);
    return pyramid[index];
  }

 private:
  int levels;                    // 图像金字塔层级
  std::vector<cv::Mat> pyramid;  // 图像 vector 序列: 最底层为裁剪后的原始图像,
                                 // 最高层为缩放 1<<levels 倍的图像

  /// @brief 读取图像以初始化图像金字塔
  /// @param src 输入图像矩阵
  /// @param levels 金字塔最高层数
  void buildPyramid(const cv::Mat &src, int levels = 0) {
    cv::Mat curImg = src.clone();

    // 裁剪图像到合适尺寸
    int suit_size = 1 << levels;
    int n_rows = (curImg.rows / suit_size) * suit_size;
    int n_cols = (curImg.cols / suit_size) * suit_size;
    curImg = curImg(cv::Rect(0, 0, n_cols, n_rows));

    // 初始化图像金字塔
    pyramid.push_back(curImg);

    // 构建图像金字塔
    for (int i = 0; i < levels - 1; i++) {
      cv::Mat downsampledImg;
      cv::pyrDown(pyramid[i], downsampledImg);
      pyramid.push_back(downsampledImg);
    }
  }
};

/// @brief 形状日志生成器
class shapeInfo_producer {
 public:
  float angle_range[2];  // 角度范围
  float scale_range[2];  // 缩放系数范围
  float angle_step;      // 角度步长
  float scale_step;      // 缩放系数步长
  float eps;             // 搜索精度

  /// @brief 日志结构体, 含旋转角和缩放系数两个参数
  struct Info {
    float angle;
    float scale;
    Info(float input_angle, float input_scale)
        : angle(input_angle), scale(input_scale) {}
  };

  /// @brief 初始化形状日志生成器的公有成员
  shapeInfo_producer() {
    angle_range[0] = angle_range[1] = 0.0f;
    scale_range[0] = scale_range[1] = 1.0f;
    angle_step = 0.0f;
    scale_step = 0.0f;
    eps = 1e-7f;
  }

  /// @brief 用构造函数读入模板图像 src 和掩图 mask, 布尔变量 padding 默认为
  /// true 表示当前读入的图像已经扩充好的 0 边界,
  /// 否则在构造函数中使用默认方法以扩充图像边界。
  /// @param input_src 待读入的模板图像 src
  /// @param input_mask 待读入的掩图 mask, 默认为与 src 等大小的全 255 像素矩阵
  /// @param padding 边界扩张选项，默认为 true
  shapeInfo_producer(const cv::Mat &input_src, cv::Mat input_mask = cv::Mat(),
                     bool padding = true) {
    CV_Assert(input_src.size() == input_mask.size());
    CV_Assert(!input_mask.empty() || input_mask.type() == CV_8U);
    // 当前图像已经扩充 0 边界
    if (padding) {
      src = input_src;
      if (!input_mask.empty())
        mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
      else
        mask = input_mask;
    } else {  // 当前图像未扩充 0 边界
      // 图像在旋转和缩放过程中有效像素到图像中心的最远距离
      int border_max = 1 + 2 * (int)sqrt(input_src.rows * input_src.rows +
                                         input_src.cols * input_src.cols);
      // 扩充边界
      cv::copyMakeBorder(input_src, src, border_max - input_src.rows / 2,
                         border_max - input_src.rows / 2,
                         border_max - input_src.cols / 2,
                         border_max - input_src.cols / 2, cv::BORDER_CONSTANT);

      if (!input_mask.empty())
        mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
      else  // 扩充掩图边界
        cv::copyMakeBorder(
            input_mask, mask, border_max - input_mask.rows / 2,
            border_max - input_mask.rows / 2, border_max - input_mask.cols / 2,
            border_max - input_mask.cols / 2, cv::BORDER_CONSTANT);
    }
  }

  /// @brief 按当前设置读入形状日志
  void produce_infos() {
    CV_Assert(scale_range[0] < scale_range[1] + eps);
    CV_Assert(angle_range[0] < angle_range[1] + eps);
    if (scale_step < eps) scale_step = 2 * eps;
    if (angle_step < eps) angle_step = 2 * eps;
    for (float scale = scale_range[0]; scale < scale_range[1] + eps;
         scale += scale_step)
      for (float angle = angle_range[0]; angle < angle_range[1] + eps;
           angle += angle_step)
        Infos.push_back(Info(angle, scale));
  }

  /// @brief 以旋转角 angle 和缩放系数 scale 对输入图像 src 作仿射变换,
  /// 返回经过仿射变换后的像素矩阵
  /// @param src 输入图像
  /// @param angle 旋转角
  /// @param scale 缩放系数
  /// @return 经过仿射变换后的矩阵
  static cv::Mat affineTrans(const cv::Mat &src, float angle, float scale) {
    cv::Mat dst;
    cv::Point2f center(cvFloor(src.cols / 2.0f), cvFloor(src.rows / 2.0f));
    cv::Mat rotate_mat = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(src, dst, rotate_mat, src.size());
    return dst;
  }

  /// @brief 按日志输出模板图像
  /// @param info 读取日志结构体
  /// @return 返回经过旋转和缩放后的模板图像矩阵
  cv::Mat src_of(const Info &info) {
    return affineTrans(src, info.angle, info.scale);
  }

  /// @brief 按日志输出掩图
  /// @param info 读取日志结构体
  /// @return 返回经过旋转和缩放后的掩图矩阵
  cv::Mat mask_of(const Info &info) {
    return affineTrans(mask, info.angle, info.scale);
  }

 private:
  std::vector<shapeInfo_producer::Info> Infos;  // 日志列表
  cv::Mat src;                                  // 模板图像矩阵
  cv::Mat mask;                                 // 模板对应掩图矩阵
};


/// @todo 完成模板 template 类
class Templates {
 public:
  Templates() {}

  void init_costable() {
    int maxValue = std::numeric_limits<ushort>::short ::max();
    cos_table = std::vector<float>(16 * maxValue);
    float maxCos;
    for (int i = 1; i <= maxValue; i++) {
      maxCos = 0.0f;
      for (int j = 0; j < 16; j++) {
        for (int k = 0; k < 16; k++) {
          if ((ushort(i)) & (1 << k))
            maxCos = maxCos < abs(cos(bit2angle(1 << k) - bit2angle(1 << j)))
                         ? abs(cos(bit2angle(1 << k) - bit2angle(1 << j)))
                         : maxCos;
        }
      }
      cos_table[i * 16 + j] = maxCos;
    }
  }

  inline static ushort angle2bit(const float &angle) {
    float angle_mod = angle > 180 ? angle - 180 : angle;
    ushort quantized = 0;
    for (int i = 0; i < 16; i++) {
      if (angle_mod <= (float)((i + 1) * 180.0f / 16.0f)) {
        quantized = (1 << i);
        break;
      }
    }
    return quantized;
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

 private:
  struct Veclist {
    cv::Point p_xy;
    ushort angle_bit;  // 16位
    TemplateVeclist(cv::Point xy, T angle) : p_xy(xy), angle_bit(angle) {}
  };

  cv::Mat src;
  cv::Mat mask;
  std::vector<float> cos_table;
  cv::Range angle_range;
  cv::Range scale_range;
  int angle_step;
  int scale_step;
};

/// @todo 完成检测算子类
class Detector {
 public:
  Detector() {}

  void private:
};

#endif  // OPENCV_LINE_2D_HPP