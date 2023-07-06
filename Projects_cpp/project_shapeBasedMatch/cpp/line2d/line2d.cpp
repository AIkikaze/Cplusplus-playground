#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;

Mat background;

#pragma region debug

void __onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    Mat *image = static_cast<cv::Mat *>(userdata);
    if (image != nullptr && !image->empty()) {
      if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
        auto pixel = image->at<ushort>(y, x);
        std::cout << "Pixel value at (" << x << ", " << y << "): " << pixel
                  << std::endl;
      }
    }
  }
}

inline Vec3b labelcolor(int label) {
  static Vec3b colors[16];

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

  return colors[label];
}

void colormap(const Mat &quantized, Mat &dst) {
  dst = Mat::zeros(quantized.size(), CV_8UC3);
  for (int r = 0; r < dst.rows; r++) {
    const ushort *quad_r = quantized.ptr<ushort>(r);
    Vec3b *dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; c++) {
      for (int k = 0; k < 16; k++)
        if (quad_r[c] & (1 << k))
          dst_r[c] = labelcolor(k);
    }
  }
}

#pragma endregion // debug
#pragma region Struct_ShapeTemplate

Ptr<ShapeTemplate> ShapeTemplate::relocate(float new_scale,
                                           float new_angle) const {
  Ptr<ShapeTemplate> ptp =
      makePtr<ShapeTemplate>(pyramid_level, new_scale, new_angle);

  double scaleFactor = new_scale / scale;
  double angleFactor = new_angle - angle;
  if (abs(scaleFactor - 1.0f) < line2d_eps && abs(angleFactor) < line2d_eps) {
    ptp->box = box;
    ptp->features = features;
    return ptp;
  }

  // 对矩形选框进行旋转缩放
  Point2f vertices[4];
  Point2f dstPoints[3];
  box.points(vertices);
  Mat rotate_mat = getRotationMatrix2D(box.center, angleFactor, scaleFactor);

  // 只用选取 3 个顶点进行变换
  for (int i = 0; i < 3; i++) {
    double new_x = rotate_mat.at<double>(0, 0) * vertices[i].x +
                   rotate_mat.at<double>(0, 1) * vertices[i].y +
                   rotate_mat.at<double>(0, 2);
    double new_y = rotate_mat.at<double>(1, 0) * vertices[i].x +
                   rotate_mat.at<double>(1, 1) * vertices[i].y +
                   rotate_mat.at<double>(1, 2);
    dstPoints[i] = Point2f(new_x, new_y);
  }

  // 更新矩形选框
  ptp->box = RotatedRect(dstPoints[0], dstPoints[1], dstPoints[2]);

  // 对特征点序列进行旋转缩放
  vector<Gradient> &relocated_featrues = ptp->features;
  auto it = features.begin(), it_end = features.end();

  for (; it != it_end; it++) {
    double new_x = rotate_mat.at<double>(0, 0) * (*it).x +
                   rotate_mat.at<double>(0, 1) * (*it).y;
    double new_y = rotate_mat.at<double>(1, 0) * (*it).x +
                   rotate_mat.at<double>(1, 1) * (*it).y;
    float new_it_angle = fmod((*it).angle - angleFactor + 360.0f, 360.0f);
    relocated_featrues.emplace_back(static_cast<int>(new_x + 0.5),
                                    static_cast<int>(new_y + 0.5),
                                    new_it_angle);
  }

  return ptp;
}

void ShapeTemplate::show_in(Mat &background, Point new_center) {
  static int base_size = 8;

  // 平移旋转矩形到新的中心点位置
  RotatedRect translatedRect = box;
  Point2f &center = translatedRect.center;
  if (new_center.x > 0 && new_center.y > 0)
    center = new_center;
  // 增加边框
  translatedRect.size.width += 10;
  translatedRect.size.height += 10;
  // 在图像上绘制旋转矩形
  Point2f vertices[4];
  translatedRect.points(vertices);
  for (int i = 0; i < 4; i++) {
    line(background, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 1);
  }

  for (const auto &f : features) {
    drawMarker(background, Point(f.x + center.x, f.y + center.y),
               labelcolor(f.label), MARKER_SQUARE, base_size >> pyramid_level);
  }
  drawMarker(background, center, Scalar(0, 255, 0), MARKER_DIAMOND,
             base_size >> pyramid_level);
}

void ShapeTemplate::show_in(Mat &background, vector<LinearMemory> &score_maps,
                            Point new_center) {
  static int base_size = 8;

  // 平移旋转矩形到新的中心点位置
  RotatedRect translatedRect = box;
  Point2f &center = translatedRect.center;
  if (new_center.x > 0 && new_center.y > 0)
    center = new_center;
  // 增加边框
  translatedRect.size.width += 10;
  translatedRect.size.height += 10;
  // 在图像上绘制旋转矩形
  Point2f vertices[4];
  translatedRect.points(vertices);
  for (int i = 0; i < 4; i++) {
    line(background, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 1);
  }

  // 定义自定义热力图的颜色映射
  std::vector<cv::Scalar> heatmapColors = {
      Scalar(128, 128, 128), // 灰色
      Scalar(255, 0, 0),     // 蓝色
      Scalar(255, 128, 0),   // 浅蓝色
      Scalar(255, 255, 0),   // 青色
      Scalar(0, 255, 0),     // 绿色
      Scalar(0, 255, 255),   // 黄色
      Scalar(0, 128, 255),   // 橙色
      Scalar(0, 0, 255),     // 红色
      Scalar(0, 0, 128)      // 深红色
  };

  for (const auto &f : features) {
    const ushort &score_at_f =
        score_maps[f.label].linear_at(f.y + center.y, f.x + center.x);
    drawMarker(background, Point(f.x + center.x, f.y + center.y),
               heatmapColors[score_at_f], MARKER_SQUARE,
               base_size >> pyramid_level);
  }
  drawMarker(background, center, Scalar(0, 255, 0), MARKER_DIAMOND,
             base_size >> pyramid_level);
}

void cropTemplate(ShapeTemplate &templ) {
  vector<Point2f> points(templ.features.size());
  for (int i = 0; i < (int)points.size(); i++)
    points[i] = Point2f(templ.features[i].x, templ.features[i].y);

  templ.box = minAreaRect(points);
  int offset_x = templ.box.center.x + 0.5f;
  int offset_y = templ.box.center.y + 0.5f;
  templ.box.center = Point2f(offset_x, offset_y);

  for (int i = 0; i < (int)templ.features.size(); i++) {
    Gradient &point = templ.features[i];
    point.x -= offset_x;
    point.y -= offset_y;
  }
}

#pragma endregion // ShapeTemplate
#pragma region Class_ColorGradientPyramid

// ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
//                                            float _magnitude_threshold,
//                                            int _count_kernel_size,
//                                            size_t _num_features)
//     : magnitude_threshold(_magnitude_threshold),
//       count_kernel_size(_count_kernel_size), num_features(_num_features),
//       pyramid_level(0), src(_src), mask(_mask) {
//   cout << "自定义构造函数" << endl;
//   update();
// }

void selectScatteredFeatures(const vector<Candidate> &candidates,
                             vector<Gradient> &features, size_t num_features,
                             float distance) {
  features.clear();
  float distance_sq = distance * distance;
  int i = 0;

  while (features.size() < num_features) {
    const Candidate &c = candidates[i];
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; j++) {
      Feature &fj = features[j];
      keep = (c.x - fj.x) * (c.x - fj.x) + (c.y - fj.y) * (c.y - fj.y) >=
             distance_sq;
    }
    if (keep)
      features.push_back(c);

    if (++i == (int)candidates.size()) {
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
}

bool ColorGradientPyramid::extractTemplate(ShapeTemplate &templ) const {
  Mat local_mask;
  if (!mask.empty()) {
    erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    subtract(mask, local_mask, local_mask);
  }

  vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  for (int r = 0; r < magnitude.rows; r++) {
    for (int l = 0; l < magnitude.cols; l++) {
      if (no_mask || mask.at<uchar>(r, l)) {
        const float &angle_at_rl = angle.at<float>(r, l);
        const float &magnitude_at_rl = magnitude.at<float>(r, l);
        const ushort &quantized_at_rl = quantized_angle.at<ushort>(r, l);
        if (quantized_at_rl > 0 && magnitude_at_rl > magnitude_threshold) {
          candidates.push_back(Candidate(l, r, angle_at_rl, magnitude_at_rl));
        }
      }
    }
  }
  if (num_features > (int)candidates.size())
    return false;

  stable_sort(candidates.begin(), candidates.end());

  float distance = static_cast<float>(candidates.size() / num_features + 1);
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  cropTemplate(templ);
  templ.pyramid_level = pyramid_level;

  return true;
}

void ColorGradientPyramid::pyrDown() {
  num_features = num_features >> 1;
  pyramid_level++;

  Size size(src.cols >> 1, src.rows >> 1);
  Mat next_src;
  resize(src, next_src, size, 0.0, 0.0, INTER_NEAREST);
  src = next_src;

  if (!mask.empty()) {
    Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
    mask = next_mask;
  }

  update();
}

void ColorGradientPyramid::affineTrans(float angle, float scale) {
  Size size = Size(src.cols * scale * 1.5f, src.rows * scale * 1.5f);
  if (mask.empty())
    mask = Mat(src.size(), CV_8U, Scalar(255));

  Point2f center(cvFloor(src.cols / 2.0f), cvFloor(src.rows / 2.0f));
  Mat rotate_mat = getRotationMatrix2D(center, angle, scale);

  // 调整旋转中心点的位置
  float offsetX = (size.width - src.cols) / 2.0f;
  float offsetY = (size.height - src.rows) / 2.0f;
  rotate_mat.at<double>(0, 2) += offsetX;
  rotate_mat.at<double>(1, 2) += offsetY;

  warpAffine(src, src, rotate_mat, size, INTER_LINEAR, BORDER_REPLICATE);

  warpAffine(mask, mask, rotate_mat, size, INTER_LINEAR, BORDER_REPLICATE);

  update();
}

static void quantizeAngle(Mat &magnitude, Mat &angle, Mat &quantized_angle,
                          float threshold, int kernel_size) {
  Mat quanized_unfiltered;
  angle.convertTo(quanized_unfiltered, CV_8U, 32.0f / 360.0f);

  for (int r = 0; r < angle.rows; r++) {
    uchar *quan_r = quanized_unfiltered.ptr(r);
    for (int c = 0; c < angle.cols; c++) {
      quan_r[c] &= 15;
    }
  }

  quantized_angle = Mat::zeros(angle.size(), CV_16U);

  int dx[kernel_size], dy[kernel_size];
  int total_kernel = kernel_size * kernel_size;
  int NEIGHBOR_THRESHOLD = total_kernel / 2 + 1;

  for (int i = 0; i < kernel_size; i++)
    dx[i] = dy[i] = -(kernel_size / 2) + i;

  for (int r = 0; r < angle.rows; r++) {
    float *mag_r = magnitude.ptr<float>(r);
    for (int c = 0; c < angle.cols; c++) {
      if (mag_r[c] <= threshold)
        continue;

      int count[16] = {0};
      int index = -1;
      int max_votes = 0;

      for (int i = 0; i < total_kernel; i++) {
        int u = r + dy[i / kernel_size], v = c + dx[i % kernel_size];
        if (u < 0 || v < 0 || u >= angle.rows || v >= angle.cols)
          continue;
        count[quanized_unfiltered.at<uchar>(u, v)]++;
      }

      for (int ori = 0; ori < QUANTIZE_BASE; ori++) {
        if (count[ori] > max_votes) {
          max_votes = count[ori];
          index = ori;
        }
      }

      if (max_votes >= NEIGHBOR_THRESHOLD)
        quantized_angle.at<ushort>(r, c) = ushort(1 << index);
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
  Scharr(smoothed, sobel_3dx, CV_32F, 1, 0, 1.0, 0, BORDER_REPLICATE);
  Scharr(smoothed, sobel_3dy, CV_32F, 0, 1, 1.0, 0, BORDER_REPLICATE);

  if (smoothed.channels() == 1) {
    sobel_dx = sobel_3dx;
    sobel_dy = sobel_3dy;
    cv::magnitude(sobel_dx, sobel_dy, magnitude);
  } else {
    for (int i = 0; i < size.height; i++) {
      for (int j = 0; j < size.width; j++) {
        float maxMagnitude = 0;
        int index = 0;
        // suitable of 3 channel img
        for (int c = 0; c < smoothed.channels(); c++) {
          float &dx = sobel_3dx.at<Vec3f>(i, j)[c];
          float &dy = sobel_3dy.at<Vec3f>(i, j)[c];
          float mag_c = dx * dx + dy * dy;
          if (mag_c > maxMagnitude) {
            maxMagnitude = mag_c;
            index = c;
          }
        }
        sobel_dx.at<float>(i, j) = sobel_3dx.at<Vec3f>(i, j)[index];
        sobel_dy.at<float>(i, j) = sobel_3dy.at<Vec3f>(i, j)[index];
        magnitude.at<float>(i, j) = maxMagnitude;
      }
    }
  }

  normalize(magnitude, magnitude, 0, 100.0f, NORM_MINMAX, CV_32F);
}

void ColorGradientPyramid::update() {
  Mat sobel_dx, sobel_dy;
  sobelMagnitude(src, magnitude, sobel_dx, sobel_dy);
  // Mat magnitude_image;
  // normalize(magnitude, magnitude_image, 0, 255, NORM_MINMAX, CV_8U);
  // namedWindow("magnitude");
  // imshow("magnitude", magnitude_image);
  // waitKey();

  phase(sobel_dx, sobel_dy, angle, true);

  quantizeAngle(magnitude, angle, quantized_angle, magnitude_threshold,
                count_kernel_size);
}

#pragma endregion // ColorGradientPyramid
#pragma region Class_TemplateSearch

struct ShapeTemplatePredicate {
  float scale;
  float angle;
  ShapeTemplatePredicate(float _scale, float _angle)
      : scale(_scale), angle(_angle) {}
  bool operator()(const Ptr<ShapeTemplate> &templ) {
    float dscale = abs(scale - templ->scale);
    float dangle =
        fmin(abs(angle - templ->angle), abs(360.0f - angle - templ->angle));
    return (dscale < 0.2f && dangle < 3.0f);
  }
};

// vector<Ptr<ShapeTemplate>> TemplateSearch::searchInRegion(float scale,
//                                                           float angle) {
//   vector<Ptr<ShapeTemplate>> target_templates;
//   copy_if(templates.begin(), templates.end(),
//   back_inserter(target_templates),
//           ShapeTemplatePredicate(scale, angle));

//   return target_templates;
// }

vector<Ptr<ShapeTemplate>> TemplateSearch::searchInRegion(float scale, float angle) {
  vector<Ptr<ShapeTemplate>> target_templates;
  
  int num_threads = omp_get_max_threads(); // 获取可用的最大线程数
  vector<vector<Ptr<ShapeTemplate>>> target_templates_per_thread(num_threads);
  
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    const int total_templates = templates.size();
    const int chunk_size = (total_templates + num_threads - 1) / num_threads; // 均分任务
    
    int start_index = thread_id * chunk_size;
    int end_index = min(start_index + chunk_size, total_templates);
    
    for (int i = start_index; i < end_index; i++) {
      Ptr<ShapeTemplate> templ = templates[i];
      
      ShapeTemplatePredicate predicate(scale, angle);
      if (predicate(templ)) {
        target_templates_per_thread[thread_id].push_back(templ);
      }
    }
  }
  
  // 合并结果
  for (const auto& templates_per_thread : target_templates_per_thread) {
    target_templates.insert(target_templates.end(), templates_per_thread.begin(), templates_per_thread.end());
  }
  
  return target_templates;
}

void TemplateSearch::build(const Search &search, ShapeTemplate &base) {
  region = search;
  rows = search.scale.values.size();
  cols = search.angle.values.size();

  const vector<float> &scales = search.scale.values;
  const vector<float> &angles = search.angle.values;

  templates.resize(rows * cols);

#pragma omp parallel for
  for (int i = 0; i < (int)scales.size(); i++) {
    for (int j = 0; j < (int)angles.size(); j++) {
      templates[i * cols + j] = base.relocate(scales[i], angles[j]);
    }
  }
}

void TemplateSearch::build(const Search &search,
                           Ptr<ColorGradientPyramid> modal) {
  region = search;
  const vector<float> &scales = search.scale.values;
  const vector<float> &angles = search.angle.values;

  rows = scales.size();
  cols = angles.size();

  templates.resize(rows * cols);

#pragma omp parallel for
  for (int i = 0; i < (int)scales.size(); i++) {
    for (int j = 0; j < (int)angles.size(); j++) {
      Ptr<ColorGradientPyramid> trans_modal(new ColorGradientPyramid(*modal));
      templates[i * cols + j] = makePtr<ShapeTemplate>(0, scales[i], angles[j]);
      trans_modal->affineTrans(angles[j], scales[i]);
      trans_modal->extractTemplate(*templates[i * cols + j]);
    }
  }

  // printf("[ template ]: scale == %.2f | angle == %.2f \n", scale, angle);
  // Mat rotate_background = trans_modal->background();
  // templates.back()->show_in(rotate_background,
  //                           templates.back()->box.center);
  // namedWindow("template", WINDOW_NORMAL);
  // imshow("template", rotate_background);
  // waitKey();
}

#pragma endregion // TemplateSearch
#pragma region Class_LinearMemory

void LinearMemory::linearize(const Mat &src) {
  rows = src.rows / block_size;
  cols = src.cols / block_size;

  create(rows * cols);

  for (int r = 0; r < rows * block_size; r++) {
    for (int c = 0; c < rows * block_size; c++) {
      this->linear_at(r, c) = src.at<uchar>(r, c);
    }
  }
}

void LinearMemory::unlinearize(Mat &dst) {
  dst.create(Size(cols * block_size, rows * block_size), CV_16S);

  for (int r = 0; r < dst.rows; r++) {
    for (int c = 0; c < dst.cols; c++) {
      dst.at<short>(r, c) = this->linear_at(r, c);
    }
  }
}

#pragma endregion // LinearMemory
#pragma region Class_Detector

static void spread(Mat &src, Mat &dst, int kernel_size) {
  CV_Assert(kernel_size >= 1);
  if (kernel_size == 1) {
    src.copyTo(dst);
    return;
  }

  int dx[kernel_size], dy[kernel_size];
  int total_kernel = kernel_size * kernel_size;

  for (int i = 0; i < kernel_size; i++)
    dx[i] = dy[i] = -(kernel_size / 2) + i;

  dst = Mat::zeros(src.size(), QUANTIZE_TYPE);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      for (int k = 0; k < total_kernel; k++) {
        int u = i + dy[k / kernel_size], v = j + dx[k % kernel_size];

        if (u < 0 || u >= src.rows || v < 0 || v >= src.cols ||
            src.at<quantize_type>(u, v) == 0)
          continue;

        dst.at<quantize_type>(i, j) |= src.at<quantize_type>(u, v);
      }
    }
  }
}

#include "similarity_lut.i"
const int bit_mask[] = {15, 240, 3840, 61440};

static void computeResponseMaps(Mat &src, vector<Mat> &response_maps) {
  response_maps.resize(QUANTIZE_BASE);
  for (int i = 0; i < QUANTIZE_BASE; i++)
    response_maps[i] = Mat::zeros(src.size(), CV_8U);

  static int bit_size = QUANTIZE_BASE / 4;
  static int lut_step = bit_size * QUANTIZE_BASE;

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (!src.at<quantize_type>(i, j))
        continue;

      // std::cout << bitset<16>(src.at<quantize_type>(i, j)) << endl;

      uchar _bit_[bit_size];
      for (int k = 0; k < bit_size; k++)
        _bit_[k] = (src.at<quantize_type>(i, j) & bit_mask[k]) >> (4 * k);

      // for (int k = 0; k < bit_size; k++)
      //   std::cout << "[" << k << "]: " << bitset<4>(_bit_[k]) << endl;

      for (int ori = 0; ori < QUANTIZE_BASE; ori++) {
        uchar maxScore = 0;

        // std::cout << "orientation: " << endl << bitset<16>(1 << ori) << endl;
        // std::cout << bitset<16>(src.at<quantize_type>(i, j)) << endl;
        for (int k = 0; k < bit_size; k++) {
          // std::cout << "[" << k << "]: " << bitset<4>(_bit_[k]) << " - " <<
          // 16 * k + (int)_bit_[k]
          //           << " -> " << (int)similarity_lut[ori * lut_step + 16 * k
          //           + (int)_bit_[k]] << endl;
          // std::cin.get();
          maxScore =
              max(maxScore,
                  similarity_lut[ori * lut_step + 16 * k + (int)_bit_[k]]);
        }

        response_maps[ori].at<uchar>(i, j) = maxScore;
      }
    }
  }
}

void Detector::setSource(Mat &src, Mat mask) {
  Ptr<ColorGradientPyramid> modal(new ColorGradientPyramid(*modality));
  modal->process(src, mask);

  for (int l = 0; l < pyramid_level; l++) {
    for (int i = 0; i < QUANTIZE_BASE; i++)
      memories_pyramid.emplace_back(block_size);

    Mat quantized, spread_quantized;
    modal->quantize(quantized);
    // Mat quantizedImage;
    // colormap(quantized, quantizedImage);
    // imshow("quantized", quantizedImage);
    // waitKey();
    // destroyWindow("quantized");

    spread(quantized, spread_quantized, spread_kernels[l]);
    // Mat spreadImage;
    // colormap(spread_quantized, spreadImage);
    // imshow("spread", spreadImage);
    // imwrite("spread.jpg", spreadImage);
    // waitKey();
    // destroyWindow("spread");

    vector<Mat> response_maps;
    computeResponseMaps(spread_quantized, response_maps);
    // for (int i = 0; i < (int)response_maps.size(); i++) {
    //   Mat response_mat;
    //   normalize(response_maps[i], response_mat, 0, 255, NORM_MINMAX, CV_8U);
    //   namedWindow("response_map", WINDOW_NORMAL);
    //   imshow("response_map", response_mat);
    //   waitKey();
    //   destroyWindow("response_map");
    // }

    for (int i = 0; i < QUANTIZE_BASE; i++)
      memories_pyramid[l * QUANTIZE_BASE + i].linearize(response_maps[i]);

    if (l != pyramid_level - 1)
      modal->pyrDown();
  }
}

  float default_scale_range[3] = {1.0f, 1.0f, 0.0f};
void Detector::setTemplate(Mat &object, Mat object_mask) {
  float default_angle_range[3] = {0.0f, 0.0f, 0.0f};

  setTemplate(object, object_mask, default_scale_range, default_angle_range);
}

void Detector::setTemplate(Mat &object, Mat object_mask,
                           const float (&scale_range)[3],
                           const float (&angle_range)[3]) {
  Search search(scale_range, angle_range);

  Ptr<ColorGradientPyramid> modal(new ColorGradientPyramid(*modality));
  modal->process(object, object_mask);

  for (int l = 0; l < pyramid_level; l++) {
    // ShapeTemplate templ(l, 1.0f, 0.0f);
    // modal->extractTemplate(templ);
    // background = modal->background();

    TemplateSearch tp_search;
    // tp_search.build(search, templ);
    tp_search.build(search, modal);

    templates_pyramid.push_back(tp_search);

    search.scale.setStep(search.scale.step * 2);
    search.angle.setStep(search.angle.step * 2);

    if (l != pyramid_level - 1)
      modal->pyrDown();
  }
}

static void computeSimilarity(LinearMemory *response_map,
                              const ShapeTemplate &templ,
                              LinearMemory &similarity) {
  similarity.create(response_map[0].linear_size(), 0);
  similarity.cols = response_map[0].cols;
  similarity.rows = response_map[0].rows;

  int size = similarity.block_size * similarity.block_size;

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    for (const auto &point : templ.features) {
      Point cur = Point(point.x + i % 4, point.y + i / 4);

      int mod_y = cur.y % 4 < 0 ? (cur.y % 4) + 4 : cur.y % 4;
      int mod_x = cur.x % 4 < 0 ? (cur.x % 4) + 4 : cur.x % 4;

      int offset =
          ((cur.y - mod_y) / 4) * similarity.cols + (cur.x - mod_x) / 4;

      // Using MIPP to optimize computations
      int j = max(0, -offset);
      mipp::Reg<short> reg_similarity;
      mipp::Reg<short> reg_response_map;
      for (; max(j, j + offset) <=
             ((int)similarity.linear_size() - mipp::N<short>());
           j += mipp::N<short>()) {
        reg_similarity.load(&similarity.at(i, j));

        reg_response_map.load(
            &response_map[point.label].at(mod_y * 4 + mod_x, j + offset));

        reg_similarity += reg_response_map;

        reg_similarity.store(&similarity.at(i, j));
      }

      for (; j < (int)similarity.linear_size(); j++)
        similarity.at(i, j) +=
            response_map[point.label].at(mod_y * 4 + mod_x, j + offset);
    }
  }
}

static Point addLocalSimilarity(LinearMemory *response_map,
                                const ShapeTemplate &templ,
                                LinearMemory &similarity, int x, int y) {
  int size = similarity.block_size * similarity.block_size;
  int offset_x = x / similarity.block_size - 2;
  int offset_y = y / similarity.block_size - 2;
  int begin_idx = offset_y * response_map->cols + offset_x;

  // printf("(%d, %d) -> (%d, %d) -> (%d, %d)\n", x, y, offset_x + 4, offset_y +
  // 4, offset_x, offset_y);

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    for (const auto &point : templ.features) {
      Point cur = Point(point.x + i % 4, point.y + i / 4);

      int mod_y = cur.y % 4 < 0 ? (cur.y % 4) + 4 : cur.y % 4;
      int mod_x = cur.x % 4 < 0 ? (cur.x % 4) + 4 : cur.x % 4;

      int offset =
          ((cur.y - mod_y) / 4) * response_map->cols + (cur.x - mod_x) / 4;
      offset += begin_idx;

      auto map_ptr = &response_map[point.label].at(mod_y * 4 + mod_x, offset);

      int offset_j = 0;
      for (int j = 0; j < (int)similarity.linear_size(); j++) {
        offset_j = j % 4 + (j / similarity.cols) * response_map->cols;
        // printf("(%d, %d) <- (%d, %d) : %hd\n", j % 8, j / 8, (offset_j +
        // offset) % response_map->cols, (offset_j + offset) /
        // response_map->cols, map_ptr[offset_j]);
        similarity.at(i, j) += map_ptr[offset_j];
        // getchar();
      }
    }
  }

  return Point(offset_x * similarity.block_size,
               offset_y * similarity.block_size);
}

// Used to filter out weak matches
struct MatchPredicate {
  float threshold;
  MatchPredicate(float _threshold) : threshold(_threshold) {}
  bool operator()(const Match &m) { return m.similarity < threshold; }
};

static void nmsMatchPoints(vector<Match> &match_points, float threshold) {
  if (match_points.empty()) return;

  stable_sort(match_points.begin(), match_points.end());

  vector<Match> filtered_points;

  for (int i = 0; i < (int)match_points.size(); i++) {
    bool keep = true;
    // ShapeTemplate *temp_i = match_points[i].templ;

    for (int j = i + 1; j < (int)match_points.size(); j++) {
      // ShapeTemplate *temp_j = match_points[j].templ;
      // 计算当前点与比较点之间的差异
      float dx = match_points[i].x - match_points[j].x;
      float dy = match_points[i].y - match_points[j].y;
      // float dscale = temp_i->scale - temp_j->scale;
      // float dangle = temp_i->angle - temp_j->angle;

      // 计算距离或重叠度量
      float distance = std::sqrt(dx * dx + dy * dy) / 2;
      // float diffScale = std::abs(dscale) * 0.02;
      // float diffAngle = std::abs(dangle) * 0.2;

      // 根据阈值判断是否保留当前点
      if (distance < threshold) {
        keep = false;
        break;
      }
    }

    if (keep)
      filtered_points.push_back(match_points[i]);
  }

  match_points = filtered_points;
}

void Detector::match(float score_threshold) {
  CV_Assert(!templates_pyramid.empty());
  CV_Assert(!memories_pyramid.empty());
  matches.clear();

  TimeCounter time1("GLOBAL"), time2("LOCAL");

  int first_match = pyramid_level - 1;
  TemplateSearch &templates = templates_pyramid[first_match];
  int num_templates = templates.size();

  for (int id = 0; id < num_templates; id++) {
    Ptr<ShapeTemplate> templ = templates[id];
    LinearMemory *response_map_begin =
        &memories_pyramid[first_match * QUANTIZE_BASE];

    LinearMemory similarity(block_size);
    time1.begin();
    computeSimilarity(response_map_begin, *templ, similarity);
    time1.countOnce();
    // Mat similarityMat, similarityImage;
    // similarity.unlinearize(similarityMat);
    // normalize(similarityMat, similarityImage, 0, 255, NORM_MINMAX, CV_8U);
    // namedWindow("similarity");
    // imshow("similarity", similarityImage);
    // setMouseCallback("similarity", __onMouse, &similarityMat);
    // waitKey();
    // destroyWindow("similarity");

    int num_features = templ->features.size();
    int raw_threshold =
        static_cast<int>((score_threshold / 100.0f) * (8 * num_features));

    for (int r = 0; r < similarity.rows * similarity.block_size; r++) {
      for (int c = 0; c < similarity.cols * similarity.block_size; c++) {
        int raw_score = similarity.linear_at(r, c);
        if (raw_score > raw_threshold) {
          float score = (raw_score * 100.0f) / (8 * num_features);
          matches.push_back(Match(c, r, score, templ));
        }
      }
    }
  }

  nmsMatchPoints(matches, 32.0f / (2 * pyramid_level));

  for (int l = first_match - 1; l >= 0; l--) {
    LinearMemory *response_map_begin = &memories_pyramid[l * QUANTIZE_BASE];

    for (int i = 0; i < (int)matches.size(); i++) {
      Match &point = matches[i];
      float &scale = point.templ->scale;
      float &angle = point.templ->angle;

      // cout << "scale -> " << scale << endl;
      // cout << "angle -> " << angle << endl;

      vector<Ptr<ShapeTemplate>> matched_templates =
          templates_pyramid[l].searchInRegion(scale, angle);

      int x = point.x * 2;
      int y = point.y * 2;

      vector<Match> candidates;

      for (int id = 0; id < (int)matched_templates.size(); id++) {
        Ptr<ShapeTemplate> templ = matched_templates[id];
        // cout << "template [" << id + 1 << "]" << endl;
        // cout << "scale -> " << templ->scale << endl;
        // cout << "angle -> " << templ->angle << endl;
        int num_features = templ->features.size();

        LinearMemory local_similarity(block_size);
        local_similarity.create(16, 0);
        local_similarity.rows = 4;
        local_similarity.cols = 4;
        // vector<bool> visited(local_similarity.size(), false);

        time2.begin();
        Point base = addLocalSimilarity(response_map_begin, *templ,
                                        local_similarity, x, y);
        time2.countOnce();
        // Mat similarityMat, similarityImage;
        // local_similarity.unlinearize(similarityMat);
        // normalize(similarityMat, similarityImage, 0, 255, NORM_MINMAX,
        // CV_8U); namedWindow("similarity", WINDOW_NORMAL);
        // imshow("similarity", similarityImage);
        // printf(" ******** scale -> %.2f angle -> %.2f ********* \n",
        // templ->scale, templ->angle); setMouseCallback("similarity",
        // __onMouse, &similarityMat); waitKey(); destroyWindow("similarity");

        int best_score = 0;
        Point best_match(-1, -1);

        for (int r = 0; r < local_similarity.rows 
                          * local_similarity.block_size; r++) {
          for (int c = 0; c < local_similarity.cols 
                            * local_similarity.block_size; c++) {
            int score = local_similarity.linear_at(r, c);
            if (score > best_score) {
              best_score = score;
              best_match = Point(c, r) + base;
            }
          }
        }

        candidates.emplace_back(best_match.x, best_match.y,
                                (best_score * 100.0f) / (8 * num_features),
                                templ);
        // point.x = best_match.x;
        // point.y = best_match.y;
        // point.similarity = (best_score * 100.0f) / (8 * num_features);
        // point.templ = templ;
        // cout << "-------- match_point --------" << endl;
        // cout << "x -> " << point.x << endl;
        // cout << "y -> " << point.y << endl;
        // cout << "similarity -> " << point.similarity << endl;
        // cout <<
        // cout << "******** match_point ********" << endl;
      }

      stable_sort(candidates.begin(), candidates.end());
      point = candidates.back();
    }

    // 使用 remove_if 结合 lambda 表达式来筛除小于阈值的元素
    matches.erase(std::remove_if(matches.begin(), matches.end(),
                  [score_threshold](Match &x) { return x.similarity < score_threshold; }),
                  matches.end());
  }

  nmsMatchPoints(matches, 4.0f);

  time1.out();
  time2.out();
}

void Detector::detectBestMatch(vector<Vec6f> &points,
                               vector<RotatedRect> &boxes) {
  points.resize(matches.size());
  boxes.resize(matches.size());

  for (int i = 0; i < (int)points.size(); i++) {
    points[i][0] = matches[i].x;
    points[i][1] = matches[i].y;
    points[i][2] = matches[i].templ->scale;
    points[i][3] = matches[i].templ->angle;
    points[i][4] = matches[i].similarity;
    boxes[i] = matches[i].templ->box;
  }
}

void Detector::draw(Mat background) {
  for (const auto &point : matches) {
    point.templ->show_in(background, memories_pyramid, Point(point.x, point.y));
    Vec3b randColor;
    randColor[0] = rand() % 155 + 100;
    randColor[1] = rand() % 155 + 100;
    randColor[2] = rand() % 155 + 100;
    putText(background, to_string(int(round(point.similarity))),
            Point(point.x - 10, point.y - 3), FONT_HERSHEY_PLAIN, 4, randColor);
  }
  if (matches.empty())
    cerr << "没有找到匹配点!" << endl;
  namedWindow("matchImage", WINDOW_NORMAL);
  imshow("matchImage", background);
  waitKey();
  destroyWindow("matchImage");
}

#pragma endregion // Detector
#pragma region ReadandWrite

void Gradient::read(const FileNode &fn) {
  x = (int)fn["x"];
  y = (int)fn["y"];
  label = (int)fn["label"];
  angle = (float)fn["angle"];
}

void Gradient::write(FileStorage &fs) const {
  fs << "{"
     << "x" << x << "y" << y << "label" << label << "angle" << angle << "}";
}

void ShapeTemplate::read(const FileNode &fn) {
  float x = (float)fn["center"][0];
  float y = (float)fn["center"][1];
  float width = (float)fn["width"];
  float height = (float)fn["height"];

  box.center = Point2f(x, y);
  box.size = Size2f(width, height);
  box.angle = (float)fn["box_angle"];

  pyramid_level = (int)fn["pyramid_level"];
  scale = (float)fn["scale"];
  angle = (float)fn["angle"];

  FileNode fn_features = fn["features"];
  features.resize(fn_features.size());
  int i = 0;
  for (const auto &f_feature : fn_features) {
    features[i++].read(f_feature);
  }
}

void ShapeTemplate::write(FileStorage &fs) const {
  fs << "center"
     << "[" << box.center.x << box.center.y << "]";

  fs << "width" << box.size.width;
  fs << "height" << box.size.height;
  fs << "box_angle" << box.angle;
  fs << "pyramid_level" << pyramid_level;
  fs << "scale" << scale;
  fs << "angle" << angle;

  fs << "features"
     << "[";
  for (int i = 0; i < (int)features.size(); i++) {
    features[i].write(fs);
  }
  fs << "]";
}

void line2d::Range::read(const FileNode &fn) {
  lower_bound = (float)fn["lower_bound"];
  upper_bound = (float)fn["upper_bound"];
  step = (float)fn["step"];
  update();
}

void line2d::Range::write(FileStorage &fs) const {
  fs << "{"
     << "lower_bound" << lower_bound << "upper_bound" << upper_bound << "step"
     << step << "}";
}

void TemplateSearch::read(const FileNode &fn) {
  fn["rows"] >> rows;
  fn["cols"] >> cols;

  region.scale.read(fn["scale_range"]);
  region.angle.read(fn["angle_range"]);

  FileNode templates_node = fn["templates"];
  CV_Assert(templates_node.type() == FileNode::SEQ);
  templates.resize(templates_node.size());
  int i = 0;
  for (auto f_template : templates_node) {
    templates[i] = makePtr<ShapeTemplate>();
    templates[i++]->read(f_template);
  }
}

void TemplateSearch::write(FileStorage &fs) const {
  fs << "rows" << rows;
  fs << "cols" << cols;

  fs << "scale_range";
  region.scale.write(fs);

  fs << "angle_range";
  region.angle.write(fs);

  fs << "templates"
     << "[";
  for (const auto &templ : templates) {
    fs << "{";
    templ->write(fs);
    fs << "}";
  }
  fs << "]";
}

void ColorGradientPyramid::read(const FileNode &fn) {
  fn["magnitude_threshold"] >> magnitude_threshold;
  fn["count_kernel_size"] >> count_kernel_size;
  fn["num_features"] >> num_features;
}

void ColorGradientPyramid::write(FileStorage &fs) const {
  fs << "magnitude_threshold" << magnitude_threshold << "count_kernel_size"
     << count_kernel_size << "num_features" << num_features;
}

void LinearMemory::read(const FileNode &fn) {
  fn["block_size"] >> block_size;
  fn["rows"] >> rows;
  fn["cols"] >> cols;

  FileNode f_memories = fn["memories"];
  memories.resize(block_size * block_size);
  auto it = f_memories.begin(), it_end = f_memories.end();
  for (int i = 0; it != it_end; it++, i++) {
    for (const auto &value : (*it)) {
      memories[i].push_back(static_cast<int>(value));
    }
  }
}

void LinearMemory::write(FileStorage &fs) const {
  fs << "block_size" << block_size;
  fs << "rows" << rows;
  fs << "cols" << cols;

  fs << "memories"
     << "[";
  for (int i = 0; i < block_size * block_size; i++) {
    fs << memories[i];
  }
  fs << "]";
}

void Detector::read(const string &filename) {
  size_t dotIndex = filename.find_last_of(".");
  CV_Assert(dotIndex != std::string::npos);
  std::string basename = filename.substr(0, dotIndex);   // 文件名部分
  std::string extension = filename.substr(dotIndex + 1); // 后缀部分

  FileStorage fs(filename, FileStorage::READ);
  CV_Assert(fs.isOpened());
  FileNode fn = fs.root();

  fn["name"] >> name;
  fn["pyramid_level"] >> pyramid_level;
  fn["block_size"] >> block_size;
  fn["spread_kernels"] >> spread_kernels;

  // 读取 modality 对象
  FileNode f_modality = fn["modality"];
  modality = makePtr<ColorGradientPyramid>();
  modality->read(f_modality);

  fs.release();

  // 读取 templates_pyramid
  FileStorage fs_templates(basename + "_templates." + extension,
                           FileStorage::READ);
  CV_Assert(fs_templates.isOpened());
  fn = fs_templates.root();

  FileNode templatesNode = fn["templates_pyramid"];
  CV_Assert(templatesNode.type() == FileNode::SEQ);
  templates_pyramid.resize(templatesNode.size());
  int i = 0;
  for (auto f_template : templatesNode) {
    templates_pyramid[i++].read(f_template);
  }

  fs_templates.release();

  // 读取 memories_pyramid
  FileStorage fs_memories(basename + "_memories." + extension,
                          FileStorage::READ);
  CV_Assert(fs_memories.isOpened());
  fn = fs_memories.root();

  FileNode memoriesNode = fn["memories_pyramid"];
  CV_Assert(memoriesNode.type() == FileNode::SEQ);
  memories_pyramid.resize(memoriesNode.size());
  i = 0;
  for (auto f_memory : memoriesNode) {
    memories_pyramid[i++].read(f_memory);
  }

  fs_memories.release();
}

void Detector::write(const string &filename) const {
  size_t dotIndex = filename.find_last_of(".");
  CV_Assert(dotIndex != std::string::npos);
  std::string basename = filename.substr(0, dotIndex);   // 文件名部分
  std::string extension = filename.substr(dotIndex + 1); // 后缀部分

  FileStorage fs(filename, FileStorage::WRITE);
  CV_Assert(fs.isOpened());

  fs << "name" << name;
  fs << "pyramid_level" << pyramid_level;
  fs << "block_size" << block_size;
  fs << "spread_kernels" << spread_kernels;

  // 写入 modality 对象
  fs << "modality"
     << "{";
  modality->write(fs);
  fs << "}";

  fs.release();

  FileStorage fs_templates(basename + "_templates." + extension,
                           FileStorage::WRITE);
  CV_Assert(fs_templates.isOpened());
  fs_templates << "templates_pyramid"
               << "[";
  for (int i = 0; i < (int)templates_pyramid.size(); i++) {
    fs_templates << "{";
    templates_pyramid[i].write(fs_templates);
    fs_templates << "}";
  }
  fs_templates << "]";
  fs_templates.release();

  FileStorage fs_memories(basename + "_memories." + extension,
                          FileStorage::WRITE);
  CV_Assert(fs_memories.isOpened());
  fs_memories << "memories_pyramid"
              << "[";
  for (int i = 0; i < (int)memories_pyramid.size(); i++) {
    fs_memories << "{";
    memories_pyramid[i].write(fs_memories);
    fs_memories << "}";
  }
  fs_memories << "]";
  fs_memories.release();
}

#pragma endregion // ReadandWrite