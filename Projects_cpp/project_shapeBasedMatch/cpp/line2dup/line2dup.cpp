#include "line2dup.hpp"
using namespace line2Dup;
using namespace std;
using namespace cv;

/// struct Feature

void Feature::read(const FileNode &fn) {
  FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const {
  fs << "[:" << x << y << label << "]";
}

// struct Template

Rect cropTemplate(vector<Template> &templates) {
  int min_x = numeric_limits<int>::max();
  int max_x = numeric_limits<int>::min();
  int min_y = numeric_limits<int>::max();
  int max_y = numeric_limits<int>::min();
  int highest_level = 0;

  // First: find min/max feature x,y over all pyramid levelds and modalities
  for (int i = 0; i < (int)templates.size(); i++) {
    Template &templ = templates[i];
    for (int j = 0; j < (int)templ.features.size(); j++) {
      // rescaling the position to the lowest level in pyramid
      int x = templ.features[j].x << templ.pyramid_level;
      int y = templ.features[j].y << templ.pyramid_level;
      // comparing
      min_x = min(min_x, x);
      max_x = max(max_x, x);
      min_y = min(min_y, y);
      max_y = max(max_y, y);
      // keep the maximum level
      highest_level = max(highest_level, templ.pyramid_level);
    }
  }

  // Notice: x = ~.x << pyramid_level is always even except pryramid_level == 0
  // and (min_x, min_y) should be the refrence point of the template (_x_, _y_)
  // so we shold better let (min_x, min_y) uniquely corespond to one point in
  // all level
  /// @example: (min_x, min_y) == (11, 5), highest_level == 3
  ///                         at level     0     1     2     3
  /// min_x : 11 -> (11 >> 3) << 3 = 8     8 <-> 4 <-> 2 <-> 1
  /// min_y : 5  -> (5 >> 3) << 3  = 0     0 <-> 0 <-> 0 <-> 0
  min_x = (min_x >> highest_level) << highest_level;
  min_y = (min_y >> highest_level) << highest_level;

  // Second: set width / height and shift all feature positions
  for (int i = 0; i < (int)templates.size(); i++) {
    Template &templ = templates[i];
    templ.width = (max_x - min_x) >> templ.pyramid_level;
    templ.height = (max_x - min_x) >> templ.pyramid_level;
    int offset_x = min_x >> templ.pyramid_level;
    int offset_y = min_y >> templ.pyramid_level;

    for (int j = 0; j < (int)templ.features.size(); j++) {
      /// @todo: why "-=" offset
      templ.features[j].x -= offset_x;
      templ.features[j].y -= offset_y;
    }
  }

  return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Template::read(const FileNode &fn) {
  width = fn["width"];
  height = fn["height"];
  pyramid_level = fn["pyramid_level"];

  FileNode featrues_fn = fn["features"];
  features.resize(featrues_fn.size());
  FileNodeIterator it = featrues_fn.begin(), it_end = featrues_fn.end();
  for (int i = 0; it != it_end; i++, it++) {
    features[i].read(*it);
  }
}

void Template::write(FileStorage &fs) const {
  fs << "width" << width;
  fs << "height" << height;
  fs << "pyramid_level" << pyramid_level;

  fs << "features"
     << "[";
  for (int i = 0; i < (int)features.size(); i++) {
    features[i].write(fs);
  }
  fs << "]";
}

// class Modality

void QuantizedPyramid::selectScatteredFeatures(
    const vector<Candidate> &candidates, vector<Feature> &features,
    size_t num_features, float distance) {
  features.clear();
  float distance_sq = distance * distance;
  int i = 0;

  while (features.size() < num_features) {
    const Candidate &c = candidates[i];
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; j++) {
      Feature &fj = features[j];
      keep =
          (c.f.x - fj.x) * (c.f.x - fj.x) + (c.f.y - fj.y) * (c.f.y - fj.y) >=
          distance_sq;
    }
    if (keep)
      features.push_back(c.f);

    if (++i == (int)candidates.size()) {
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
}

Ptr<Modality> Modality::create(const String &modality_type) {
  if (modality_type == "ColorGradient")
    return makePtr<ColorGradient>();
  else
    return Ptr<Modality>();
}

Ptr<Modality> Modality::create(const FileNode &fn) {
  String type = fn["type"];
  Ptr<Modality> modality = create(type);
  modality->read(fn);
  return modality;
}

void line2Dup::colormap(const Mat &quantized, Mat &dst) {
  Vec3b colors[16];

  // Set 16 distinct color parameters
  colors[0] = Vec3b(255, 0, 0);      // Blue
  colors[1] = Vec3b(0, 255, 0);      // Green
  colors[2] = Vec3b(0, 0, 255);      // Red
  colors[3] = Vec3b(255, 255, 0);    // Cyan
  colors[4] = Vec3b(255, 0, 255);    // Magenta
  colors[5] = Vec3b(0, 255, 255);    // Yellow
  colors[6] = Vec3b(128, 0, 0);      // Dark Blue
  colors[7] = Vec3b(0, 128, 0);      // Dark Green
  colors[8] = Vec3b(0, 0, 128);      // Dark Red
  colors[9] = Vec3b(128, 128, 0);    // Olive
  colors[10] = Vec3b(128, 0, 128);   // Dark Magenta
  colors[11] = Vec3b(0, 128, 128);   // Dark Yellow
  colors[12] = Vec3b(192, 192, 192); // Silver
  colors[13] = Vec3b(128, 128, 128); // Gray
  colors[14] = Vec3b(255, 165, 0);   // Orange
  colors[15] = Vec3b(128, 0, 0);     // Brown

  dst = Mat::zeros(quantized.size(), CV_8UC3);
  for (int r = 0; r < dst.rows; r++) {
    const quantize_type *quad_r = quantized.ptr<quantize_type>(r);
    Vec3b *dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; c++) {
      for (int k = 0; k < 16; k++)
        if (quad_r[c] & (1 << k))
          dst_r[c] = colors[k];
    }
  }
}

// #include <opencv2/imgproc.hpp>
void line2Dup::drawFeatures(InputOutputArray img, const vector<Template> &templates,
                  const Point2i &_p_, int size) {
#ifdef HAVE_OPENCV_IMGPROC
  static Scalar colors[] = {{0, 0, 255}, {0, 255, 0}};
  static int markers[] = {MARKER_SQUARE, MARKER_DIAMOND};

  int modality = 0;
  for (const Template &t : templates) {
    if (t.pyramid_level != 0)
      continue;
    for (const Feature &f : t.features) {
      drawMarker(img, _p_ + Point(f.x, f.y), colors[int(modality != 0)],
                 markers[int(modality != 0)], size);
    }
    modality++;
  }
#else
  CV_Error(Error::StsAssert, "need improc module");
#endif
}

/// Color Gradient Modality

static const char MOD_NAME[] = "ColorGradient";

ColorGradient::ColorGradient()
    : weak_threshold(10.0f), strong_threshold(55.0f), num_features(63) {}

ColorGradient::ColorGradient(float _weak_threshold, float _strong_threshold,
                             size_t _num_features)
    : weak_threshold(_weak_threshold), strong_threshold(_strong_threshold),
      num_features(_num_features) {}

Ptr<ColorGradient> ColorGradient::create(float _weak_threshold,
                                         float _strong_threshold,
                                         size_t _num_features) {
  return makePtr<ColorGradient>(_weak_threshold, _strong_threshold,
                                _num_features);
}

String ColorGradient::name() const { return MOD_NAME; }

void ColorGradient::read(const FileNode &fn) {
  String type = fn["type"];
  CV_Assert(type == MOD_NAME);

  weak_threshold = fn["weak_threshold"];
  strong_threshold = fn["strong_threshold"];
  num_features = int(fn["num_features"]);
}

void ColorGradient::write(FileStorage &fs) const {
  fs << "type" << MOD_NAME;
  fs << "weak_threshold" << weak_threshold;
  fs << "strong_threshold" << strong_threshold;
  fs << "num_features" << int(num_features);
}

Ptr<QuantizedPyramid> ColorGradient::processTempl(const cv::Mat &src,
                                                  const cv::Mat &mask) const {
  return makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features,
                                       strong_threshold);
}

/// Color Gradient Pyramid

inline int angle2label(const float &alpha) {
  int quantized_alpha = int(alpha * (2 * QUANTIZE_BASE) / 360.0);
  return quantized_alpha & (QUANTIZE_BASE - 1);
}

inline int angle2label(const quantize_type &alpha) {
  for (int i = 0; i < QUANTIZE_BASE; i++) {
    if (alpha & (1 << i))
      return i;
  }
  return 0;
}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold,
                                           float _strong_threshold,
                                           size_t _num_features)
    : pyramid_level(0), src(_src), mask(_mask), weak_threshold(_weak_threshold),
      strong_threshold(_strong_threshold), num_features(_num_features) {
  update();
}

void ColorGradientPyramid::quantize(Mat &dst) const {
  dst = Mat::zeros(angle.size(), CV_8U);
  angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const {
  Mat local_mask;
  if (!mask.empty()) {
    erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    subtract(mask, local_mask, local_mask);
  }

  vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  float threshold_sq = strong_threshold * strong_threshold;
  for (int r = 0; r < magnitude.rows; r++) {
    for (int l = 0; l < magnitude.cols; l++) {
      if (no_mask || mask.at<uchar>(r, l)) {
        const quantize_type &angle_at_rl = angle.at<quantize_type>(r, l);
        const float &magnitude_at_rl = magnitude.at<float>(r, l);
        if (angle_at_rl > 0 && magnitude_at_rl > threshold_sq) {
          candidates.push_back(
              Candidate(l, r, angle2label(angle_at_rl), magnitude_at_rl));
        }
      }
    }
  }
  if (candidates.size() < num_features)
    return false;

  stable_sort(candidates.begin(), candidates.end());

  float distance = static_cast<float>(candidates.size() / num_features + 1);
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  templ.width = -1;
  templ.height = -1;
  templ.pyramid_level = pyramid_level;

  return true;
}

void ColorGradientPyramid::pyrDown() {
  num_features = num_features >> 2;
  pyramid_level++;

  Size size(src.cols >> 1, src.rows >> 1);
  Mat next_src;
  cv::resize(src, next_src, size, 0.0, 0.0, INTER_NEAREST);
  src = next_src;

  if (!mask.empty()) {
    Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
    mask = next_mask;
  }

  update();
}

static void quantizeAngle(Mat &magnitude, Mat &angle, Mat &quantized_angle,
                          float threshold) {
  Mat quanized_unfiltered;
  angle.convertTo(quanized_unfiltered, CV_8U, 32.0 / 360.0);
  
  for (int r = 0; r < angle.rows; r++) {
    uchar *quan_r = quanized_unfiltered.ptr(r);
    for (int c = 0; c < angle.cols; c++) {
      quan_r[c] &= (QUANTIZE_BASE - 1);
    }
  }

  quantized_angle = Mat::zeros(angle.size(), QUANTIZE_TYPE);

  int dx[3], dy[3];
  for (int i = 0; i < 3; i++) 
    dx[i] = dy[i] = -1 + i;

  for (int r = 0; r < angle.rows; r++) {
    float *mag_r = magnitude.ptr<float>(r);
    for (int c = 0; c < angle.cols; c++) {
      if (mag_r[c] <= threshold) continue;
      int count[16] = {0};
      int index = -1;
      int max_votes = 0;
      for (int i = 0; i < 9; i++) {
        int u = r + dy[i/3], v = c + dx[i%3];
        if (u < 0 || v < 0 || u >= angle.rows || v >= angle.cols) continue;
        int cur_label = quanized_unfiltered.at<uchar>(u, v);
        if (++count[cur_label] > max_votes) {
          max_votes = count[cur_label];
          index = cur_label;
        }
      }
      static int NEIGHBOR_THRESHOLD = 5;
      if (max_votes >= NEIGHBOR_THRESHOLD) 
        quantized_angle.at<quantize_type>(r, c) = (1 << index);
    }
  }
}

static void sobelMagnitude(const Mat &src, Mat &magnitude, Mat &sobel_dx,
                           Mat &sobel_dy) {
  // Allocate temporary buffers
  Size size = src.size();
  Mat sobel_3dx;
  Mat sobel_3dy;
  Mat smoothed;
  // Initialize in/out params
  sobel_dx.create(size, CV_32F);
  sobel_dy.create(size, CV_32F);
  magnitude.create(size, CV_32F);

  // Calculate the magnitude matrix of 3/1 channel img
  static const int KERNEL_SIZE = 7;
  GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0,
               BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0, BORDER_REPLICATE);

  for (int i = 0; i < size.height; i++) {
    for (int j = 0; j < size.width; j++) {
      int maxMagnitude = 0;
      int index = 0;
      // suitable of 3 or 1 channel img
      for (int c = 0; c < smoothed.channels(); c++) {
        short &dx = sobel_3dx.at<Vec3s>(i, j)[c];
        short &dy = sobel_3dy.at<Vec3s>(i, j)[c];
        int mag_c = dx * dx + dy * dy;
        if (mag_c > maxMagnitude) {
          maxMagnitude = mag_c;
          index = c;
        }
      }
      sobel_dx.at<float>(i, j) = sobel_3dx.at<Vec3s>(i, j)[index];
      sobel_dy.at<float>(i, j) = sobel_3dy.at<Vec3s>(i, j)[index];
      magnitude.at<float>(i, j) = maxMagnitude;
    }
  }
}

void ColorGradientPyramid::update() {
  Mat sobel_dx, sobel_dy;
  sobelMagnitude(src, magnitude, sobel_dx, sobel_dy);
  // Mat mag_display;
  // normalize(magnitude, mag_display, 0, 255, NORM_MINMAX, CV_8U);
  // namedWindow("magnitude", WINDOW_NORMAL);
  // imshow("magnitude", mag_display);
  // waitKey();

  Mat sobel_ag;
  phase(sobel_dx, sobel_dy, sobel_ag, true);
  // Mat angle_display;
  // normalize(sobel_ag, angle_display, 0, 255, NORM_MINMAX, CV_8U);
  // namedWindow("quantized", WINDOW_NORMAL);
  // imshow("quantized", angle_display);
  // waitKey();

  quantizeAngle(magnitude, sobel_ag, angle,
                strong_threshold * strong_threshold);
  // Mat quantized;
  // colormap(angle, quantized);
  // namedWindow("quantized", WINDOW_NORMAL);
  // imshow("quantized", quantized);
  // waitKey();

  // destroyAllWindows();
}

/// Response Maps

static void orUnaligned16u(const quantize_type *src, int src_stride,
                           quantize_type *dst, int dst_stride, int width,
                           int height) {
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

static void spread(const Mat &src, Mat &dst, int T) {
  dst = Mat::zeros(src.size(), QUANTIZE_TYPE);
  for (int r = 0; r < T; r++) {
    for (int l = 0; l < T; l++) {
      orUnaligned16u(&src.at<quantize_type>(r, l),
                     static_cast<int>(src.step1()), dst.ptr<quantize_type>(),
                     static_cast<int>(dst.step1()), src.cols - l, src.rows - r);
    }
  }
}

/*
@code : create_similarity_lut.py
--- code begin
similarity_lut = [0] * 1024  # 创建一个长度为 1024 的数组，并初始化为 0

for ori in range(16):
    for j in range(64):
        bit = (j % 16) << ((j // 16) * 4)
        for k in range(9):
            base = (1 << ((ori + k) % 16)) | (1 << ((ori - k) % 16))
            if base & bit:
                similarity_lut[ori * 64 + j] = 8 - k
                break

# 将结果输出为 C++ 格式的数组
print("uchar similarity_lut[1024] = {", end='')
for i in range(1024):
    print(f"{similarity_lut[i]}", end=(", " if i < 1023 else ''))
print("};")
--- code end
*/

#include "similariry_lut.i"

static void computeResponseMaps(const Mat &src, vector<Mat> &response_maps) {
  CV_Assert(src.rows % QUANTIZE_BASE == 0);
  CV_Assert(src.cols % QUANTIZE_BASE == 0);

  response_maps.resize(QUANTIZE_BASE);
  for (int i = 0; i < QUANTIZE_BASE; i++)
    response_maps[i].create(src.size(), QUANTIZE_TYPE);

#if QUANTIZE_BASE == 8
  Mat lsb(src.size(), CV_8U);
  Mat msb(src.size(), CV_8U);

  for (int r = 0; r < src.rows; r++) {
    const quantize_type *src_r = src.ptr<quantize_type>(r);
    uchar *lsb_r = lsb.ptr<uchar>(r);
    uchar *msb_r = msb.ptr<uchar>(r);

    l_mask = 15;
    m_mask = 240;

    for (int c = 0; c < src.cols; c++) {
      lsb_r[c] = src_r[c] & 15;
      msb_r[c] = (src_r[c] & 240) >> 4;
    }
  }

  for (int ori = 0; ori < QUANTIZE_BASE; ori++) {
    quantize_type *map_data = response_maps[ori].ptr<quantize_type>();
    uchar *lsb_data = lsb.ptr<uchar>();
    uchar *msb_data = msb.ptr<uchar>();

    const uchar *lut_low = similarity_lut + 32 * ori;
    const uchar *lut_hi = lut_low + 16;

    for (int i = 0; i < src.rows * src.cols; i++) {
      map_data[i] = max(lut_low[lsb_data[i]], lut_hi[msb_data[i]]);
    }
  }
#elif QUANTIZE_BASE == 16

  vector<Mat> base(QUANTIZE_BASE / 4);
  for (int i = 0; i < base.size(); i++)
    base[i].create(src.size(), CV_8U);

  for (int r = 0; r < src.rows; r++) {
    const quantize_type *src_r = src.ptr<quantize_type>(r);
    uchar *base_r[4];
    int mask[4] = {15, 240, 3840, 61440};

    for (int i = 0; i < 4; i++) {
      base_r[i] = base[i].ptr(r);
      for (int c = 0; c < src.cols; c++) {
        base_r[i][c] = (src_r[c] & mask[i]) >> (i * 4);
      }
    }
  }

  for (int ori = 0; ori < QUANTIZE_BASE; ori++) {
    quantize_type *map_data = response_maps[ori].ptr<quantize_type>();
    uchar *base_data[4];
    for (int i = 0; i < 4; i++)
      base_data[i] = base[i].ptr<uchar>();

    const uchar *lut_data[4];
    for (int i = 0; i < 4; i++)
      lut_data[i] = similarity_lut + 64 * ori + (i * 16);

    uchar base_max;
    for (int i = 0; i < src.rows * src.cols; i++) {
      base_max = 0;
      for (int j = 0; j < 4; j++)
        base_max = max(lut_data[j][base_data[j][i]], base_max);
      map_data[i] = base_max;
    }
  }
#endif
}

static void linearize(const Mat &response_map, Mat &linearized, int T) {
  CV_Assert(response_map.rows % T == 0);
  CV_Assert(response_map.cols % T == 0);

  int mem_width = response_map.cols / T;
  int mem_height = response_map.rows / T;
  linearized.create(T * T, mem_width * mem_height, CV_8U);

  for (int i = 0; i < T * T; i++) {
    uchar *memory = linearized.ptr<uchar>(i);
    for (int r = 0; r < mem_height; r++) {
      const uchar *response_data = response_map.ptr<uchar>(r * T + i / T);
      for (int c = 0; c < mem_width; c++) {
        memory[r * mem_width + c] = response_data[c * T + i % T];
      }
    }
  }
}

/// Linearized Similarities

static const unsigned char *
accessLinearMemory(const vector<Mat> &linear_memories, const Feature &f, int T,
                   int W) {
  // Retrieve the TxT grid of linear memories associated with the feature label
  const Mat &memory_grid = linear_memories[f.label];
  CV_DbgAssert(memory_grid.rows == T * T);
  CV_DbgAssert(f.x >= 0);
  CV_DbgAssert(f.y >= 0);
  // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of
  // memory_grid)
  int grid_x = f.x % T;
  int grid_y = f.y % T;
  int grid_index = grid_y * T + grid_x;
  CV_DbgAssert(grid_index >= 0);
  CV_DbgAssert(grid_index < memory_grid.rows);
  const unsigned char *memory = memory_grid.ptr(grid_index);
  // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM,
  // the input image width decimated by T.
  int lm_x = f.x / T;
  int lm_y = f.y / T;
  int lm_index = lm_y * W + lm_x;
  CV_DbgAssert(lm_index >= 0);
  CV_DbgAssert(lm_index < memory_grid.cols);
  return memory + lm_index;
}

static void similarity(const vector<Mat> &linear_memories,
                       const Template &templ, Mat &dst, Size size, int T) {
  int W = size.width / T;
  int H = size.height / T;

  int wf = (templ.width - 1) / T + 1;
  int hf = (templ.height - 1) / T + 1;
  int templ_position = (H - hf) * W + (W - wf);

  dst = Mat::zeros(H, W, CV_16U);
  ushort *dst_data = dst.ptr<ushort>();

  for (int i = 0; i < (int)templ.features.size(); i++) {
    Feature fi = templ.features[i];

    if (fi.x < 0 || fi.y >= size.width || fi.y < 0 || fi.y >= size.height)
      continue;

    const uchar *lm_data = accessLinearMemory(linear_memories, fi, T, W);

    for (int j = 0; j < templ_position; j++)
      dst_data[j] += lm_data[j];
  }
}

static void similarityLocal(const vector<Mat> &linear_memories,
                            const Template &templ, Mat &dst, Size size, int T,
                            Point center) {
  int W = size.width / T;
  int offset_x = (center.x / T - 8) * T;
  int offset_y = (center.y / T - 8) * T;

  dst = Mat::zeros(16, 16, CV_16U);
  ushort *dst_data = dst.ptr<ushort>();

  for (int i = 0; i < (int)templ.features.size(); i++) {
    Feature fi = templ.features[i];
    fi.x += offset_x;
    fi.y += offset_y;

    if (fi.x < 0 || fi.y >= size.width || fi.y < 0 || fi.y >= size.height)
      continue;

    const uchar *lm_data = accessLinearMemory(linear_memories, fi, T, W);

    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 16; c++) {
        dst_data[c] += lm_data[c];
      }
      dst_data += 16;
      lm_data += W;
    }
  }
}

static void addUnaligned8u16u(ushort *res, const uchar *src, int length) {
  const ushort *end = res + length;

  while (res != end) {
    *res += *src;
    ++res;
    ++src;
  }
}

static void addSimilarities(const vector<Mat> &similarities, Mat &dst) {
  if (similarities.size() == 1) {
    similarities[0].convertTo(dst, CV_16U);
  } else {
    dst = Mat::zeros(similarities[0].size(), CV_16U);
    for (int i = 0; i < (int)similarities.size(); i++)
      addUnaligned8u16u(dst.ptr<ushort>(), similarities[i].ptr(),
                        static_cast<int>(dst.total()));
  }
}

/// Detector API

Detector::Detector(const vector<Ptr<Modality>> &_modalities,
                   const vector<int> &T_pyramid)
    : modalities(_modalities),
      pyramid_levels(static_cast<int>(T_pyramid.size())),
      T_at_level(T_pyramid) {}

void Detector::match(const vector<Mat> &sources, float threshold,
                     vector<Match> &matches, const vector<String> &class_ids,
                     OutputArrayOfArrays quantized_images,
                     const vector<Mat> &masks) const {
  matches.clear();

  if (quantized_images.needed())
    quantized_images.create(
        1, static_cast<int>(pyramid_levels * modalities.size()), CV_8U);

  CV_Assert(sources.size() == modalities.size());

  // 从 modalities 中读取源图像和遮罩
  vector<Ptr<QuantizedPyramid>> quantizers;
  for (int i = 0; i < (int)modalities.size(); i++) {
    Mat mask, source = sources[i];
    if (!masks.empty()) {
      CV_Assert(masks.size() == modalities.size());
      mask = masks[i];
    }
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modalities[i]->process(source, mask));
  }

  // lm[i][j][k] : 以 Mat 为元素的三维数组
  // i -> pyramid_level; j -> index of modalities; k -> orientation label
  LinearMemoryPyramid lm_pyramid(
      pyramid_levels,
      vector<LinearMemories>(modalities.size(), LinearMemories(QUANTIZE_BASE)));

  /// @todo 为什么要记录矩阵大小？
  vector<Size> sizes;
  for (int l = 0; l < pyramid_levels; l++) {
    /// @todo T_at_level 是做什么用的？
    int T = T_at_level[l];

    vector<LinearMemories> &lm_level = lm_pyramid[l];

    if (l > 0)
      for (int i = 0; i < (int)quantizers.size(); i++)
        quantizers[i]->pyrDown();

    Mat quantized, spread_quantized;
    vector<Mat> response_maps;
    for (int i = 0; i < (int)quantizers.size(); i++) {
      quantizers[i]->quantize(quantized);
      spread(quantized, spread_quantized, T);
      computeResponseMaps(spread_quantized, response_maps);

      LinearMemories &memories = lm_level[i];
      for (int j = 0; j < QUANTIZE_BASE; j++)
        linearize(response_maps[j], memories[j], T);

      if (quantized_images.needed())
        quantized.copyTo(quantized_images.getMatRef(
            static_cast<int>(l * quantizers.size() + i)));
    }
    /// @todo 为什么在这里存入 quantized 矩阵的大小
    sizes.push_back(quantized.size());
  }

  if (class_ids.empty()) {
    // 用所有模板进行匹配
    TemplateMap::const_iterator it = class_templates.begin(),
                                it_end = class_templates.end();
    for (; it != it_end; it++)
      matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
  } else {
    // 指定 class_id 进行模板匹配
    for (int i = 0; i < (int)class_ids.size(); i++) {
      TemplateMap::const_iterator it = class_templates.find(class_ids[i]);
      if (it != class_templates.end()) {
        matchClass(lm_pyramid, sizes, threshold, matches, it->first,
                   it->second);
      }
    }
  }

  sort(matches.begin(), matches.end());
  vector<Match>::iterator new_end = unique(matches.begin(), matches.end());
  matches.erase(new_end, matches.end());
}

struct MatchPredicate {
  float threshold;
  MatchPredicate(float _threshold) : threshold(_threshold) {}
  bool operator()(const Match &m) { return m.similarity < threshold; }
};

void Detector::matchClass(
    const LinearMemoryPyramid &lm_pyramid, const vector<Size> &sizes,
    float threshold, vector<Match> &matches, const String &class_id,
    const vector<TemplatePyramid> &template_pyramids) const {

  for (size_t template_id = 0; template_id < template_pyramids.size();
       template_id++) {
    // 提取当前 id 对应的模板金字塔
    const TemplatePyramid &tp = template_pyramids[template_id];

    // 提取最高层的线性存储器 -> 里边有已经计算完成的梯度响应矩阵
    const vector<LinearMemories> &lowest_lm = lm_pyramid.back();

    // 在金字塔最高层为每个模型计算相似度矩阵
    vector<Mat> similarities(modalities.size());

    int lowest_start = static_cast<int>(tp.size() - modalities.size());
    int lowest_T = T_at_level.back();
    int num_features = 0;

    for (int i = 0; i < (int)modalities.size(); i++) {
      const Template &templ = tp[lowest_start + i];
      num_features += static_cast<int>(templ.features.size());
      similarity(lowest_lm[i], templ, similarities[i], sizes.back(), lowest_T);
    }

    Mat total_similarity;
    addSimilarities(similarities, total_similarity);

    int raw_threshold = static_cast<int>(
        2 * num_features + (threshold / 100.0f) * (2 * num_features) + 0.5f);

    vector<Match> candidates;
    for (int r = 0; r < total_similarity.rows; r++) {
      ushort *row = total_similarity.ptr<ushort>(r);
      for (int c = 0; c < total_similarity.cols; c++) {
        int raw_score = row[c];
        if (raw_score > raw_threshold) {
          int offset = lowest_T / 2 + (lowest_T % 2 - 1);
          int x = c * lowest_T + offset;
          int y = r * lowest_T + offset;
          float score = (raw_score * 100.0f) / (4 * num_features) + 0.5f;
          candidates.push_back(
              Match(x, y, score, class_id, static_cast<int>(template_id)));
        }
      }
    }

    for (int l = pyramid_levels - 2; l >= 0; l--) {
      const vector<LinearMemories> &lms = lm_pyramid[l];
      int T = T_at_level[l];
      int start = static_cast<int>(l * modalities.size());
      Size size = sizes[l];
      int border = 8 * T;
      int offset = T / 2 + (T % 2 - 1);
      int max_x = size.width - tp[start].width - border;
      int max_y = size.height - tp[start].height - border;

      vector<Mat> similarities2(modalities.size());
      Mat total_similarity2;
      for (int m = 0; m < (int)candidates.size(); m++) {
        Match &match2 = candidates[m];
        int x = match2.x * 2 + 1;
        int y = match2.y * 2 + 1;

        x = max(x, border);
        y = max(y, border);

        x = min(x, max_x);
        y = min(y, max_y);

        int numFeatures = 0;
        for (int i = 0; i < (int)modalities.size(); i++) {
          const Template &templ = tp[start + i];
          numFeatures += static_cast<int>(templ.features.size());
          similarityLocal(lms[i], templ, similarities2[i], size, T,
                          Point(x, y));
        }
        addSimilarities(similarities2, total_similarity2);

        int best_score = 0;
        int best_r = -1, best_c = -1;
        for (int r = 0; r < total_similarity2.rows; r++) {
          ushort *row = total_similarity2.ptr<ushort>(r);
          for (int c = 0; c < total_similarity2.cols; c++) {
            int score = row[c];
            if (score > best_score) {
              best_score = score;
              best_r = r;
              best_c = c;
            }
          }
        }
        match2.x = (x / T - 8 + best_c) * T + offset;
        match2.y = (y / T - 8 + best_r) * T + offset;
        match2.similarity = (best_score * 100.0f) / (4 * numFeatures);
      }

      vector<Match>::iterator new_end = remove_if(
          candidates.begin(), candidates.end(), MatchPredicate(threshold));
      candidates.erase(new_end, candidates.end());
    }
    matches.insert(matches.begin(), candidates.begin(), candidates.end());
  }
}

int Detector::addTemplate(const vector<Mat> &sources,
                          const String &class_id, const Mat &object_mask,
                          Rect *bounding_box) {
  int num_modalities = static_cast<int>(modalities.size());
  vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());

  TemplatePyramid tp;
  
  tp.resize(num_modalities * pyramid_levels);

  for (int i = 0; i < num_modalities; i++) {
    // 存入 num_modalities 个模板突袭那个，并初始化
    Ptr<QuantizedPyramid> qp = modalities[i]->process(sources[i], object_mask);
    // 在金字塔各层级中提取特征，提取模板
    for (int l = 0; l < pyramid_levels; l++) {
      if (l > 0)
        qp->pyrDown();
      bool success = qp->extractTemplate(tp[l * num_modalities + i]);
      if (!success) return -1;
    }
  }

  // 得到模板所在区域
  Rect bb = cropTemplate(tp);
  if (bounding_box) *bounding_box = bb;

  template_pyramids.push_back(tp);
  return template_id;
}

int Detector::addSynthicTemplate(const vector<Template> &templates,
                                 const String &class_id) {
  vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());
  template_pyramids.push_back(templates);
  return template_id;
}

const vector<Template> &Detector::getTemplates(const String &class_id, 
                                         int template_id) const {
  TemplateMap::const_iterator it = class_templates.find(class_id);
  CV_Assert(it != class_templates.end());
  CV_Assert(it->second.size() > size_t(template_id));
  return it->second[template_id];
}

int Detector::numTemplates() const {
  int res = 0;
  TemplateMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
  for (; it != it_end; it++)
    res += static_cast<int>(it->second.size());
  return res;
}

int Detector::numTemplates(const String &class_id) const {
  TemplateMap::const_iterator it = class_templates.find(class_id);
  if (it == class_templates.end()) return 0;
  return static_cast<int>(it->second.size());
}

vector<String> Detector::classIds() const {
  vector<String> ids;
  TemplateMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
  for (; it != it_end; it++)
    ids.push_back(it->first);
  return ids;
}

void Detector::read(const FileNode &fn) {
  class_templates.clear();
  pyramid_levels = fn["pyramid_levels"];
  fn["T"] >> T_at_level;

  modalities.clear();
  FileNode modalities_fn = fn["modailities"];
  FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
  modalities.resize(modalities_fn.size());
  for (; it != it_end; it++) {
    modalities.push_back(Modality::create(*it));
  }
}

void Detector::write(FileStorage &fs) const {
  fs << "pyramid_levels" << pyramid_levels;
  fs << "T" << T_at_level;
  fs << "modalities" << "[:";
  for (int i = 0; i < (int)modalities.size(); i++) {
    fs << "{";
    modalities[i]->write(fs);
    fs << "}";
  }
  fs << "]"; 
}

String Detector::readClass(const FileNode &fn,
                           const String &class_id_override) {
  FileNode mod_fn = fn["modalities"];
  CV_Assert(mod_fn.size() == modalities.size());
  FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
  for (int i = 0; mod_it != mod_it_end; mod_it++, i++)
    CV_Assert(modalities[i]->name() == (String)(*mod_it));
  CV_Assert((int)fn["pyramid_levels"] == pyramid_levels);

  String class_id;
  if (class_id_override.empty()) {
    String class_id_tmp = fn["class_id"];
    CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
    class_id = class_id_tmp;
  }
  else {
    class_id = class_id_override;
  }

  TemplateMap::value_type v(class_id, vector<TemplatePyramid>());
  vector<TemplatePyramid> &tps = v.second;

  FileNode tps_fn = fn["template_pyramids"];
  tps.resize(tps_fn.size());
  FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
  for (int expected_id = 0; tps_it != tps_it_end; tps_it++, expected_id++) {
    int template_id = (*tps_it)["template_id"];
    CV_Assert(template_id == expected_id);
    FileNode templates_fn = (*tps_it)["templates"];
    tps[template_id].resize(templates_fn.size());

    FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
    for (int i = 0; templ_it != templ_it_end; templ_it++, i++) 
      tps[template_id][i].read(*templ_it);
  }

  class_templates.insert(v);
  return class_id;
}

void Detector::writeClass(const String &class_id,
                          FileStorage &fs) const {
  TemplateMap::const_iterator it = class_templates.find(class_id);
  CV_Assert(it != class_templates.end());
  const vector<TemplatePyramid> &tps = it->second;

  fs << "class_id" << it->first;
  fs << "modalities" << "[:";

  for (int i = 0; i < (int)modalities.size(); i++)
    fs << modalities[i]->name();

  fs << "]";
  fs << "pyramid_levels" << pyramid_levels;
  fs << "template_pyramids" << "[";
  for (int i = 0; i < (int)tps.size(); i++) {
    const TemplatePyramid &tp = tps[i];
    fs << "{";
    fs << "template_id" << (int)i;
    fs << "templates" << "[";
    for (int j = 0; j < (int)tp.size(); j++) {
      fs << "{";
      tp[j].write(fs);
      fs << "}"; // current template
    }
    fs << "]"; // templates
    fs << "}"; // current pyramid
  }
  fs << "]"; // pyramids
}

void Detector::readClass(const vector<String> &class_ids,
                         const String &format) {
  for (int i = 0; i < (int)class_ids.size(); i++) {
    const String &class_id = class_ids[i];
    String filename = cv::format(format.c_str(), class_id.c_str());
    FileStorage fs(filename, FileStorage::WRITE);
    writeClass(class_id, fs);
  }
}

static const int T_DEFAULTS[] = {5, 8};

Ptr<Detector> line2Dup::getDefaultLINE() {
  vector< Ptr<Modality> > modalities;
  modalities.push_back(makePtr<ColorGradient>());
  return makePtr<Detector>(modalities, vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}
