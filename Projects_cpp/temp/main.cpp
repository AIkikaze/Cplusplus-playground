#include "MIPP/mipp.h"
#include "header.hpp"
#include <bitset>
#include <iostream>
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using std::cin;
using std::cout;

#define QUANTIZE_BASE 16
#define eps 1e-5

double time_or = 0;
double time_mipp_or = 0;

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

Timer _time_;

// simd信息获取输出
void PrintOpenCVInfo() {
  std::cout << "--------------------------OpenCV "
               "informaintion--------------------------"
            << std::endl;
  // OpenCV版本，Universal Simd在3.4.x版本就开始提供，不过建议使用4.x版本
  std::cout << "OpenCV version:" << cv::getVersionString() << std::endl;
  std::cout << "Simd info: " << std::endl;
#ifdef CV_SIMD
  // 是否支持simd优化
  std::cout << "CV_SIMD : " << CVAUX_STR(CV_SIMD) << std::endl;
  // simd优化内存位宽
  std::cout << "CV_SIMD_WIDTH : " << CVAUX_STR(CV_SIMD_WIDTH) << std::endl;
  // 128bit位宽优化是否支持，绝大部分x86架构cpu支持，如常用的sse指令集
  std::cout << "CV_SIMD128 : " << CVAUX_STR(CV_SIMD128) << std::endl;
  // 256bit位宽优化是否支持，大部分近几年的x86架构cpu支持， avxz指令集
  std::cout << "CV_SIMD256: " << CVAUX_STR(CV_SIMD256) << std::endl;
  // 512bit位宽是否支持，这个intel最先搞得，近几年的intel cpu与最新的AMD锐龙支持
  std::cout << "CV_SIMD512 : " CVAUX_STR(CV_SIMD512) << std::endl;
#else
  std::cout << "CV_SIMD is NOT defined." << std::endl;
#endif

#ifdef CV_SIMD
  std::cout << "sizeof(v_uint8) = " << sizeof(cv::v_uint8) << std::endl;
  std::cout << "sizeof(v_int32) = " << sizeof(cv::v_int32) << std::endl;
  std::cout << "sizeof(v_float32) = " << sizeof(cv::v_float32) << std::endl;
#endif
}

void MIPP_test() {
  cout << "MIPP tests" << endl;
  cout << "----------" << endl << endl;

  cout << "Instr. type:       " << mipp::InstructionType << endl;
  cout << "Instr. full type:  " << mipp::InstructionFullType << endl;
  cout << "Instr. version:    " << mipp::InstructionVersion << endl;
  cout << "Instr. size:       " << mipp::RegisterSizeBit << " bits" << endl;
  cout << "Instr. lanes:      " << mipp::Lanes << endl;
  cout << "64-bit support:    " << (mipp::Support64Bit ? "yes" : "no") << endl;
  cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no")
       << endl;

#ifndef has_max_int8_t
  cout << "in this SIMD, int8 max is not inplemented by MIPP" << endl;
#endif

#ifndef has_shuff_int8_t
  cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << endl;
#endif

  cout << "----------" << endl << endl;
}

inline int angle2label(const float &alpha) {
  int quantized_alpha = int(alpha * (2 * QUANTIZE_BASE) / 360.0);
  return quantized_alpha & (QUANTIZE_BASE - 1);
}

void debug(ushort *src, int src_stride, int width, int height) {
  cout << "[";
  for (int r = 0; r < height; r++) {
    for (int l = 0; l < width; l++) {
      cout << bitset<16>(*(src + l));
      if (l < width - 1)
        cout << ", ";
    }
    src += src_stride;
    cout << endl;
  }
  cout << "]" << endl;
}

void orUnaligned16u(ushort *src, int src_stride, ushort *dst, int dst_stride,
                    int width, int height) {
  for (int r = 0; r < height; r++) {
    int l = 0;

    // 处理未对齐的部分
    while ((reinterpret_cast<unsigned long long>(src + l) % 16) != 0) {
      dst[l] |= src[l];
      l++;
    }

    int _l_ = l;

    // 使用 SIMD 指令进行按位或运算
    // mipp::N<uchar>() -> 16
    for (l <<= 1; l < 2 * width; l += mipp::N<uchar>()) {
      mipp::Reg<uchar> src_v((uchar *)src + l);
      mipp::Reg<uchar> dst_v((uchar *)dst + l);

      mipp::Reg<uchar> res_v = mipp::orb(src_v, dst_v);
      res_v.store((uchar *)dst + l);

      _l_ += 8;
      // debug(ini_dst, dst_stride, width, height);
      // cin.get();
    }

    for (_l_ -= 8; _l_ < width; _l_++)
      dst[_l_] |= src[_l_];

    // 移动到下一行的内存区域
    src += src_stride;
    dst += dst_stride;
  }
}

Mat bitwiseOr(const Mat &mat1, const Mat &mat2) {
  Mat result;
  bitwise_or(mat1, mat2, result);
  return result;
}

Mat bitwiseOrMIPP(Mat &mat1, Mat &mat2) {
  Mat result = mat2.clone();
  orUnaligned16u(mat1.ptr<ushort>(), static_cast<int>(mat1.step1()),
                 result.ptr<ushort>(), static_cast<int>(result.step1()),
                 result.cols, result.rows);
  return result;
}

Mat bitwiseOrbyPixel(Mat &mat1, Mat &mat2) {
  Mat result;
  result.create(mat1.size(), CV_16U);

  for (int i = 0; i < result.rows; i++) {
    ushort *res_p = result.ptr<ushort>(i);
    ushort *mat1_p = mat1.ptr<ushort>(i);
    ushort *mat2_p = mat2.ptr<ushort>(i);

    for (int j = 0; j < result.cols; j++) {
      res_p[j] = mat1_p[j] | mat2_p[j];
    }
  }

  return result;
}

void compareResults(Mat &mat1, Mat &mat2) {
  _time_.reset();
  Mat result1 = bitwiseOr(mat1, mat2); // 使用第一种方法计算按位或运算
  time_or += _time_.elapsed();

  _time_.reset();
  Mat result2 = bitwiseOrbyPixel(mat1, mat2);
  // Mat result2 = bitwiseOrMIPP(mat1, mat2); // 使用第二种方法计算按位或运算
  time_mipp_or += _time_.elapsed();

  if (countNonZero(result1 != result2) > 0) {
    std::cerr << "Results are different!" << endl;
  } else {
    // cout << "Results are the same." << endl;
  }
}

void COMPARE_test() {
  int times = 1000;
  int t = 0;

  RNG rng; // 随机数生成器
  Mat mat1, mat2;

  while (++t <= times) {
    rng.state = getTickCount(); // 使用当前系统时间作为随机数生成器的种子

    int rows = 1000; // 随机生成行数
    int cols = 1000; // 随机生成列数

    mat1.create(Size(cols, rows), CV_16U);
    mat2.create(Size(cols, rows), CV_16U);

    rng.fill(mat1, RNG::UNIFORM, 0, 65535); // 随机填充第一个矩阵
    rng.fill(mat2, RNG::UNIFORM, 0, 65535); // 随机填充第二个矩阵

    // cout << "test [" << t << "] ";
    // cout << "compare in mat [" << rows << " x " << cols << "]: ";

    compareResults(mat1, mat2); // 比较两种方法的结果
  }

  cout << "time_or: " << time_or << "s" << endl;
  cout << "time_mipp_or: " << time_mipp_or << "s" << endl;
}

int main() {
  PrintOpenCVInfo();
  MIPP_test();

  COMPARE_test();

  return 0;
}