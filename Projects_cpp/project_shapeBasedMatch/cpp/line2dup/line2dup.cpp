#include "line2dup.hpp"
using namespace cv;
using namespace std;
using namespace line2Dup;

Mat background;

/// debug
void __onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    cv::Mat *image = static_cast<cv::Mat *>(userdata);
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

/// struct Gradient

void Gradient::read(const FileNode &fn) {
  FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label >> angle;
}

void Gradient::write(FileStorage &fs) const {
  fs << "[:" << x << y << label << angle << "]";
}

/// class ShapeTemplate

void ShapeTemplate::read(const FileNode &fn) {
  Point2f center(fn["center"][0], fn["center"][1]);
  Size2f size(fn["width"], fn["height"]);
  float angle(fn["angle"]);
  box = RotatedRect(center, size, angle);
  pyramid_level = fn["pyramid_level"];

  FileNode featrues_fn = fn["features"];
  features.resize(featrues_fn.size());
  auto it = featrues_fn.begin(), it_end = featrues_fn.end();
  for (int i = 0; it != it_end; i++, it++) {
    features[i].read(*it);
  }
}

void ShapeTemplate::write(FileStorage &fs) const {
  fs << "center"
     << "[" << box.center.x << box.center.y << "]";
  fs << "width" << box.size.width;
  fs << "height" << box.size.height;
  fs << "angle" << box.angle;
  fs << "pyramid_level" << pyramid_level;

  fs << "features"
     << "[";
  for (int i = 0; i < (int)features.size(); i++) {
    features[i].write(fs);
  }
  fs << "]";
}

Ptr<ShapeTemplate> ShapeTemplate::relocate(float new_scale, float new_angle) {
  Ptr<ShapeTemplate> ptp =
      makePtr<ShapeTemplate>(pyramid_level, new_scale, new_angle);

  float scaleFactor = new_scale / scale;
  float angleFactor = new_angle - angle;
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
  vector<Gradient>::iterator it = features.begin(), it_end = features.end();

  for (; it != it_end; it++) {
    double new_x = rotate_mat.at<double>(0, 0) * (*it).x +
                   rotate_mat.at<double>(0, 1) * (*it).y;
    double new_y = rotate_mat.at<double>(1, 0) * (*it).x +
                   rotate_mat.at<double>(1, 1) * (*it).y;
    float new_it_angle = fmod((*it).angle - new_angle + 360.0f, 360.0f);
    relocated_featrues.push_back(
        Gradient(new_x + 0.5, new_y + 0.5, new_it_angle));
  }

  return ptp;
}

void ShapeTemplate::show_in(cv::Mat &background, cv::Point new_center) {
  static int base_size = 8;

  // 平移旋转矩形到新的中心点位置
  cv::RotatedRect translatedRect = box;
  Point2f &center = translatedRect.center;
  if (new_center.x > 0 && new_center.y > 0)
    center = new_center;
  // 增加边框
  translatedRect.size.width += 10;
  translatedRect.size.height += 10;
  // 在图像上绘制旋转矩形
  cv::Point2f vertices[4];
  translatedRect.points(vertices);
  for (int i = 0; i < 4; i++) {
    cv::line(background, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0),
             1);
  }

  for (const auto &f : features) {
    cv::drawMarker(background, Point(f.x + center.x, f.y + center.y),
                   labelcolor(f.label), cv::MARKER_SQUARE,
                   base_size >> pyramid_level);
  }
  cv::drawMarker(background, center, cv::Scalar(0, 255, 0), cv::MARKER_DIAMOND,
                 base_size >> pyramid_level);
}

void line2Dup::ShapeTemplate::show_in(cv::Mat &background,
                                      vector<LinearMemory> &score_maps,
                                      cv::Point new_center) {
  static int base_size = 8;

  // 平移旋转矩形到新的中心点位置
  cv::RotatedRect translatedRect = box;
  Point2f &center = translatedRect.center;
  if (new_center.x > 0 && new_center.y > 0)
    center = new_center;
  // 增加边框
  translatedRect.size.width += 10;
  translatedRect.size.height += 10;
  // 在图像上绘制旋转矩形
  cv::Point2f vertices[4];
  translatedRect.points(vertices);
  for (int i = 0; i < 4; i++) {
    cv::line(background, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0),
             1);
  }

  // 定义自定义热力图的颜色映射
  std::vector<cv::Scalar> heatmapColors = {
      cv::Scalar(32, 32, 32),  // 灰色
      cv::Scalar(255, 0, 0),   // 蓝色
      cv::Scalar(255, 128, 0), // 浅蓝色
      cv::Scalar(255, 255, 0), // 青色
      cv::Scalar(0, 255, 0),   // 绿色
      cv::Scalar(0, 255, 255), // 黄色
      cv::Scalar(0, 128, 255), // 橙色
      cv::Scalar(0, 0, 255),   // 红色
      cv::Scalar(0, 0, 128)    // 深红色
  };

  for (const auto &f : features) {
    const ushort &score_at_f =
        score_maps[f.label].linear_at(f.y + center.y, f.x + center.x);
    cv::drawMarker(background, Point(f.x + center.x, f.y + center.y),
                   heatmapColors[score_at_f], cv::MARKER_SQUARE,
                   base_size >> pyramid_level);
  }
  cv::drawMarker(background, center, cv::Scalar(0, 255, 0), cv::MARKER_DIAMOND,
                 base_size >> pyramid_level);
}

void cropTemplate(ShapeTemplate &templ) {
  vector<Point2f> points(templ.features.size());
  for (int i = 0; i < (int)points.size(); i++)
    points[i] = Point2f(templ.features[i].x, templ.features[i].y);

  templ.box = minAreaRect(points);
  int offset_x = templ.box.center.x + 0.5f;
  int offset_y = templ.box.center.y + 0.5f;

  for (int i = 0; i < (int)templ.features.size(); i++) {
    Gradient &point = templ.features[i];
    point.x -= offset_x;
    point.y -= offset_y;
  }
}

/// Templates Search

vector<Ptr<ShapeTemplate>> TemplateSearch::searchInRegion(float scale,
                                                          float angle) {
  std::vector<float> &scales = region.scale.values;
  std::vector<float> &angles = region.angle.values;

  auto y = std::lower_bound(scales.begin(), scales.end(), scale);
  auto x = std::lower_bound(angles.begin(), angles.end(), angle);

  vector<Ptr<ShapeTemplate>> target_templates;

  int r_start =
      std::max(0, static_cast<int>(std::distance(scales.begin(), y)) - 4);
  int r_end =
      std::min(rows, static_cast<int>(std::distance(scales.begin(), y)) + 5);

  int c_start =
      std::max(0, static_cast<int>(std::distance(angles.begin(), x)) - 4);
  int c_end =
      std::min(cols, static_cast<int>(std::distance(angles.begin(), x)) + 5);

  for (int r = r_start; r < r_end; r++) {
    for (int c = c_start; c < c_end; c++) {
      target_templates.push_back(templates[r * cols + c]);
    }
  }

  return target_templates;
}

void TemplateSearch::build(const Search &search, ShapeTemplate &base) {
  region = search;
  rows = search.scale.values.size();
  cols = search.angle.values.size();

  for (auto &scale : search.scale.values)
    for (auto &angle : search.angle.values)
      templates.push_back(base.relocate(scale, angle));
  // Mat rotate_mat =
  //     getRotationMatrix2D(base.box.center, angle, scale);
  // Mat rotate_background(background.rows, background.cols,
  //                       background.type());
  // cv::warpAffine(background, rotate_background, rotate_mat,
  //                rotate_background.size(), cv::INTER_LINEAR,
  //                cv::BORDER_REPLICATE);
  // templates.back()->show_in(rotate_background);
  // namedWindow("template", WINDOW_NORMAL);
  // imshow("template", rotate_background);
  // // imwrite("template.jpg", rotate_background);
  // waitKey();
  // destroyWindow("template");
}


void line2Dup::Range::read(cv::FileNode &fn) {
  fn["lower_bound"] >> lower_bound;
  fn["upper_bound"] >> upper_bound;
  fn["step"] >> step;
  update();
}

void line2Dup::Range::write(cv::FileStorage &fs) const {
  fs << "lower_bound" << lower_bound;
  fs << "upper_bound" << upper_bound;
  fs << "step" << step;
}

void TemplateSearch::read(cv::FileNode &fn) {
  fn["rows"] >> rows;
  fn["cols"] >> cols;

  cv::FileNode angle_range = fn["angle_range"], 
               scale_range = fn["scale_range"];
  region.angle.read(angle_range);
  region.scale.read(scale_range);
  
  cv::FileNode templates_node = fn["templates"];
  templates.resize(templates_node.size());
  int i = 0;
  for (auto f_template : templates_node) {
    templates[i++]->read(f_template);
  }
} 

void TemplateSearch::write(cv::FileStorage &fs) const {

}

/// class ColorGradientPyramid

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _magnitude_threshold,
                                           int _count_kernel_size,
                                           size_t _num_features)
    : pyramid_level(0), src(_src), mask(_mask),
      magnitude_threshold(_magnitude_threshold),
      count_kernel_size(_count_kernel_size), num_features(_num_features) {
  cout << "自定义构造函数" << endl;
  update();
}

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
  if (candidates.size() < num_features)
    return false;

  stable_sort(candidates.begin(), candidates.end());

  float distance = static_cast<float>(candidates.size() / num_features + 1);
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  cropTemplate(templ);
  templ.pyramid_level = pyramid_level;

  return true;
}

void ColorGradientPyramid::read(cv::FileNode &fn) {
  fn["magnitude_threshold"] >> magnitude_threshold;
  fn["count_kernel_size"] >> count_kernel_size;
  fn["num_features"] >> num_features;
} 

void ColorGradientPyramid::write(cv::FileStorage &fs) const {
  fs << "magnitude_threshold" << magnitude_threshold;
  fs << "count_kernel_size" << count_kernel_size;
  fs << "num_features" << num_features;
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

static void quantizeAngle(Mat &magnitude, Mat &angle, Mat &quantized_angle,
                          float threshold, int kernel_size) {
  Mat quanized_unfiltered;
  angle.convertTo(quanized_unfiltered, CV_8U, 32.0 / 360.0);

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
  Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0, BORDER_REPLICATE);

  if (smoothed.channels() == 1) {
    for (int i = 0; i < size.height; i++) {
      for (int j = 0; j < size.width; j++) {
        sobel_dx.at<float>(i, j) = sobel_3dx.at<ushort>(i, j);
        sobel_dy.at<float>(i, j) = sobel_3dy.at<ushort>(i, j);
      }
    }
    cv::magnitude(sobel_dx, sobel_dy, magnitude);
  } else {
    for (int i = 0; i < size.height; i++) {
      for (int j = 0; j < size.width; j++) {
        int maxMagnitude = 0;
        int index = 0;
        // suitable of 3 or 1 channel img
        for (int c = 0; c < smoothed.channels(); c++) {
          short &dx = sobel_3dx.at<cv::Vec3s>(i, j)[c];
          short &dy = sobel_3dy.at<cv::Vec3s>(i, j)[c];
          int mag_c = dx * dx + dy * dy;
          if (mag_c > maxMagnitude) {
            maxMagnitude = mag_c;
            index = c;
          }
        }
        sobel_dx.at<float>(i, j) = sobel_3dx.at<cv::Vec3s>(i, j)[index];
        sobel_dy.at<float>(i, j) = sobel_3dy.at<cv::Vec3s>(i, j)[index];
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

void LinearMemory::linearize(const cv::Mat &src) {
  int new_rows = (src.rows / block_size) * block_size;
  int new_cols = (src.cols / block_size) * block_size;
  Mat croped_src = src(Rect(0, 0, new_cols, new_rows));

  rows = croped_src.rows / block_size;
  cols = croped_src.cols / block_size;
  create(rows * cols);

  for (int r = 0; r < new_rows; r++) {
    for (int c = 0; c < new_cols; c++) {
      this->linear_at(r, c) = src.at<uchar>(r, c);
    }
  }
}

void LinearMemory::unlinearize(cv::Mat &dst) {
  dst.create(Size(cols * block_size, rows * block_size), CV_16U);

  for (int r = 0; r < dst.rows; r++) {
    for (int c = 0; c < dst.cols; c++) {
      dst.at<ushort>(r, c) = this->linear_at(r, c);
    }
  }
}

void LinearMemory::read(cv::FileNode &fn) {
  fn["block_size"] >> block_size;
  fn["rows"] >> rows;
  fn["cols"] >> cols;
  memories.resize(block_size * block_size);

  for (int i = 0; i < block_size * block_size; i++) {
    std::string key = "memory_" + std::to_string(i);
    fn[key] >> memories[i];
  }
}

void LinearMemory::write(cv::FileStorage &fs) const {
  fs << "block_size" << block_size;
  fs << "rows" << rows;
  fs << "cols" << cols;

  for (int i = 0; i < block_size * block_size; i++) {
    std::string key = "memory_" + std::to_string(i);
    fs << key << memories[i];
  }
}

/// Detector

static void spread(Mat &src, Mat &dst, int kernel_size) {
  dst = Mat::zeros(src.size(), QUANTIZE_TYPE);
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          int u = i + dy;
          int v = j + dx;
          if (u < 0 || u >= src.rows || v < 0 || v >= src.cols ||
              src.at<quantize_type>(u, v) == 0)
            continue;
          dst.at<quantize_type>(i, j) |= src.at<quantize_type>(u, v);
        }
      }
    }
  }
}

#include "similariry_lut.i"
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

void Detector::setSource(cv::Mat &src, cv::Mat mask) {
  modality = modality->process(src, mask);

  for (int l = 0; l < pyramid_level; l++) {
    for (int i = 0; i < QUANTIZE_BASE; i++)
      memories_pyramid.emplace_back(block_size);

    Mat quantized, spread_quantized;
    modality->quantize(quantized);
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
      modality->pyrDown();
  }
}

void Detector::setTemplate(cv::Mat &object, cv::Mat object_mask) {
  float default_scale_range[3] = {1.0f, 1.0f, 0.0f};
  float default_angle_range[3] = {0.0f, 0.0f, 0.0f};

  setTemplate(object, object_mask, default_scale_range, default_angle_range);
}

void Detector::setTemplate(cv::Mat &object, cv::Mat object_mask, 
                           const float (&scale_range)[3], 
                           const float (&angle_range)[3]) {
  Search search(scale_range, angle_range);

  modality = modality->process(object, object_mask);

  for (int l = 0; l < pyramid_level; l++) {
    vector<Ptr<ShapeTemplate>> templs;
    ShapeTemplate templ(l, 1.0f, 0.0f);
    modality->extractTemplate(templ);
    // background = modality->background();

    TemplateSearch tp_search;
    tp_search.build(search, templ);

    templates_pyramid.push_back(tp_search);

    search.scale.setStep(search.scale.step * 2);
    search.angle.setStep(search.angle.step * 2);

    if (l != pyramid_level - 1)
      modality->pyrDown();
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
      for (;max(j, j + offset) <= ((int)similarity.linear_size() - mipp::N<short>());
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
  if (match_points.empty())
    return;

  stable_sort(match_points.begin(), match_points.end(),
              [](const Match &a, const Match &b) {
                return a.similarity < b.similarity;
              });

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
    // Filter out any matches that drop below the similarity threshold
    std::vector<Match>::iterator new_end = std::remove_if(
        matches.begin(), matches.end(), MatchPredicate(score_threshold));
    matches.erase(new_end, matches.end());

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

        int best_score = 0;
        Point best_match(-1, -1);

        for (int r = 0; r < local_similarity.rows * local_similarity.block_size;
             r++) {
          for (int c = 0;
               c < local_similarity.cols * local_similarity.block_size; c++) {
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
  }

  nmsMatchPoints(matches, 8.0f);

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

void Detector::draw(cv::Mat background) {
  for (const auto &point : matches) {
    point.templ->show_in(background, memories_pyramid,
                         Point(point.x, point.y));
    cv::Vec3b randColor;
    randColor[0] = rand() % 155 + 100;
    randColor[1] = rand() % 155 + 100;
    randColor[2] = rand() % 155 + 100;
    cv::putText(background, to_string(int(round(point.similarity))),
                cv::Point(point.x - 10, point.y - 3), cv::FONT_HERSHEY_PLAIN, 4,
                randColor);
  }
  if (matches.empty())
    cerr << "没有找到匹配点!" << endl;
  cv::namedWindow("matchImage", cv::WINDOW_NORMAL);
  cv::imshow("matchImage", background);
  cv::waitKey();
  cv::destroyWindow("matchImage");
}

void Detector::read(cv::FileNode& fn) {
  fn["name"] >> name;
  fn["pyramid_level"] >> pyramid_level;
  fn["block_size"] >> block_size;
  fn["spread_kernels"] >> spread_kernels;

  // 读取 modality 对象
  cv::FileNode f_modality = fn["modality"];
  modality->read(f_modality);

  // 读取 templates_pyramid
  cv::FileNode templatesNode = fn["templates_pyramid"];
  CV_Assert(templatesNode.type() == FileNode::SEQ);
  templates_pyramid.resize(templatesNode.size());
  int i = 0;
  for (cv::FileNode f_template : templatesNode) {
    templates_pyramid[i++].read(f_template);
  }

  // 读取 memories_pyramid
  cv::FileNode memoriesNode = fn["memories_pyramid"];
  CV_Assert(memoriesNode.type() == FileNode::SEQ);
  memories_pyramid.resize(memoriesNode.size());
  i = 0;
  for (cv::FileNode f_memory : memoriesNode) {
    memories_pyramid[i++].read(f_memory);
  }
}

void Detector::write(cv::FileStorage& fs) const {
  fs << "name" << name;
  fs << "pyramid_level" << pyramid_level;
  fs << "block_size" << block_size;
  fs << "spread_kernels" << spread_kernels;

  // 写入 modality 对象
  fs << "modality" << "[";
  modality->write(fs);
  fs << "]";

  // 写入 templates_pyramid
  fs << "templates_pyramid" << "[";
  for (const TemplateSearch& templateSearch : templates_pyramid) {
    cv::FileStorage templateFS(fs, fs->name + "_template");
    templateSearch.write(templateFS);
    templateFS.release();
  }
  fs << "]";

  // 写入 memories_pyramid
  fs << "memories_pyramid" << "[";
  for (const LinearMemory& linearMemory : memories_pyramid) {
    cv::FileStorage memoryFS(fs, fs->name + "_memory");
    linearMemory.write(memoryFS);
    memoryFS.release();
  }
  fs << "]";
}
