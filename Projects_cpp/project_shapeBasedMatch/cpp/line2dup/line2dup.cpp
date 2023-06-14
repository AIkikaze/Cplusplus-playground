#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;

/// debug

void colormap(const Mat &quantized, Mat &dst) {
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
    const ushort *quad_r = quantized.ptr<ushort>(r);
    Vec3b *dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; c++) {
      for (int k = 0; k < 16; k++)
        if (quad_r[c] & (1 << k))
          dst_r[c] = colors[k];
    }
  }
}

/// struct Feature

void Feature::read(const FileNode &fn) {
  FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const {
  fs << "[:" << x << y << label << "]";
}

/// struct Template

void Template::read(const FileNode &fn) {
  Point2f center(fn["center"][0], fn["center"][1]);
  Size2f size(fn["width"], fn["height"]);
  float angle(fn["angle"]);
  box = RotatedRect(center, size, angle);
  pyramid_level = fn["pyramid_level"];

  FileNode featrues_fn = fn["features"];
  features.resize(featrues_fn.size());
  FileNodeIterator it = featrues_fn.begin(), it_end = featrues_fn.end();
  for (int i = 0; it != it_end; i++, it++) {
    features[i].read(*it);
  }
}

void Template::write(FileStorage &fs) const {
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

void ShapeTemplate::relocate(ShapeTemplate &templ) {
  // 对矩形选框进行旋转缩放
  RotatedRect &tb = templ.box;
  Point2f vertices[4];
  Point2f dstPoints[3];
  tb.points(vertices);
  Mat rotate_mat = getRotationMatrix2D(tb.center, templ.scale, templ.angle);

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
  RotatedRect new_box(dstPoints[0], dstPoints[1], dstPoints[2]);
  tb = new_box;

  // 对特征点序列进行旋转缩放
  vector<Gradient> relocated_featrues;
  vector<Gradient>::iterator it = templ.features.begin(),
                             it_end = templ.features.end();

  for (; it != it_end; it++) {
    double new_x = rotate_mat.at<double>(0, 0) * (*it).x +
                   rotate_mat.at<double>(0, 1) * (*it).y +
                   rotate_mat.at<double>(0, 2);
    double new_y = rotate_mat.at<double>(1, 0) * (*it).x +
                   rotate_mat.at<double>(1, 1) * (*it).y +
                   rotate_mat.at<double>(1, 2);
    float new_angle = fmod((*it).angle - templ.angle + 360.0f, 360.0f);
    relocated_featrues.push_back(Gradient(new_x, new_y, new_angle));
  }

  // 更新特征点序列
  templ.features = relocated_featrues;
}

ShapeTemplate::operator Template() const {
  Template tp;
  tp.box = box;
  tp.pyramid_level = pyramid_level;
  tp.features.resize(features.size());
  for (int i = 0; i < (int)features.size(); i++)
    tp.features[i] = features[i];
  return tp;
}

void cropTemplate(Template &templ) {
  vector<Point2f> points(templ.features.size());
  for (int i = 0; i < (int)points.size(); i++)
    points[i] = Point2f(templ.features[i].x, templ.features[i].y);

  templ.box = minAreaRect(points);
  int offset_x = templ.box.center.x;
  int offset_y = templ.box.center.y;

  for (int i = 0; i < (int)templ.features.size(); i++) {
    templ.features[i].x -= offset_x;
    templ.features[i].y -= offset_y;
  }
}

/// class ColorGradientPyramid

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _magnitude_threshold,
                                           int _count_kernel_size,
                                           size_t _num_features)
    : pyramid_level(0), src(_src), mask(_mask),
      magnitude_threshold(_magnitude_threshold),
      count_kernel_size(_count_kernel_size), num_features(_num_features) {
  update();
}

inline int bit2label(const ushort &bit) {
  for (int i = 0; i < 16; i++) {
    if (bit & (1 << i))
      return i;
  }
  return 0;
}

void selectScatteredFeatures(const vector<Candidate> &candidates,
                             vector<Feature> &features, size_t num_features,
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

bool ColorGradientPyramid::extractTemplate(Template &templ) const {
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
        const ushort &angle_at_rl = angle.at<ushort>(r, l);
        const float &magnitude_at_rl = magnitude.at<float>(r, l);
        if (angle_at_rl > 0 && magnitude_at_rl > magnitude_threshold) {
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

  cropTemplate(templ);
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
        int u = r + dy[i / 3], v = c + dx[i % 3];
        if (u < 0 || v < 0 || u >= angle.rows || v >= angle.cols)
          continue;
        int cur_label = quanized_unfiltered.at<uchar>(u, v);
        if (++count[cur_label] > max_votes) {
          max_votes = count[cur_label];
          index = cur_label;
        }
      }
      if (max_votes >= NEIGHBOR_THRESHOLD)
        quantized_angle.at<ushort>(r, c) = (1 << index);
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
  normalize(magnitude, magnitude, 0, 100.0f, NORM_MINMAX, CV_32F);
}

void ColorGradientPyramid::update() {
  Mat sobel_dx, sobel_dy;
  sobelMagnitude(src, magnitude, sobel_dx, sobel_dy);

  Mat sobel_ag;
  phase(sobel_dx, sobel_dy, sobel_ag, true);

  quantizeAngle(magnitude, sobel_ag, angle, magnitude_threshold,
                count_kernel_size);
}

void LinearMemory::linearize(cv::Mat &src) {
  CV_Assert(src.rows % block_size == 0);
  CV_Assert(src.cols % block_size == 0);

  rows = src.rows / block_size;
  cols = src.cols / block_size;

  for (int r = 0; r < src.rows; r++) {
    for (int l = 0; l < src.cols; l++) {
      int order_block = (r % block_size) * block_size + (l % block_size);
      int idx_mat = (r / block_size) * cols + (l / block_size);
      memories[order_block][idx_mat] = src.at<ushort>(r, l);
    }
  }
}

void LinearMemory::unlinearize(cv::Mat &dst) {
  dst.create(Size(rows * block_size, cols * block_size), CV_16U);

  for (int r = 0; r < dst.rows; r++) {
    for (int l = 0; l < dst.cols; l++) {
      int order_block = (r % block_size) * block_size + (l % block_size);
      int idx_mat = (r / block_size) * cols + (l / block_size);
      dst.at<ushort>(r, l) = memories[order_block][idx_mat];
    }
  }
}

void Detector::setSource(cv::Mat &src, cv::Mat mask) { 
  
}

void Detector::setTemplate(cv::Mat &object, cv::Mat object_mask) {
}