#include "line2dup.hpp"
using namespace cv;
using namespace std;
using namespace line2Dup;

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

/// class ShapeTemplate

void ShapeTemplate::read(const FileNode &fn) {
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
  Ptr<ShapeTemplate> ptp = makePtr<ShapeTemplate>(pyramid_level, scale * new_scale, angle - new_angle);
  
  if (abs(new_scale - 1.0) < line2d_eps && abs(new_angle - 0.0) < line2d_eps) {
    ptp->box = box;
    ptp->features.resize(features.size());
    for (int i = 0; i < (int)features.size(); i++)
      ptp->features[i] = features[i];
    return ptp;
  }

  // 对矩形选框进行旋转缩放
  RotatedRect &tb = ptp->box;
  Point2f vertices[4];
  Point2f dstPoints[3];
  tb.points(vertices);
  Mat rotate_mat = getRotationMatrix2D(tb.center, new_scale, new_angle);

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
  vector<Gradient> &relocated_featrues = ptp->features;
  vector<Gradient>::iterator it = features.begin(),
                             it_end = features.end();

  for (; it != it_end; it++) {
    double new_x = rotate_mat.at<double>(0, 0) * (*it).x +
                   rotate_mat.at<double>(0, 1) * (*it).y +
                   rotate_mat.at<double>(0, 2);
    double new_y = rotate_mat.at<double>(1, 0) * (*it).x +
                   rotate_mat.at<double>(1, 1) * (*it).y +
                   rotate_mat.at<double>(1, 2);
    float new_it_angle = fmod((*it).angle - new_angle + 360.0f, 360.0f);
    relocated_featrues.push_back(Gradient(new_x, new_y, new_it_angle));
  }

  return ptp;
}

void cropTemplate(ShapeTemplate &templ) {
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
        if (angle_at_rl > 0 && magnitude_at_rl > magnitude_threshold) {
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

void ColorGradientPyramid::pyrDown() {
  num_features = num_features >> 2;
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

  phase(sobel_dx, sobel_dy, angle, true);

  quantizeAngle(magnitude, angle, quantized_angle, magnitude_threshold,
                count_kernel_size);
}

void LinearMemory::linearize(cv::Mat &src) {
  int new_rows = (src.rows / block_size + 1) * block_size;
  int new_cols = (src.cols / block_size + 1) * block_size;
  Mat bordered_src;
  copyMakeBorder(src, bordered_src, 0, new_rows - src.rows, 0,
                 new_cols - src.cols, BORDER_REPLICATE);

  // CV_Assert(src.rows % block_size == 0);
  // CV_Assert(src.cols % block_size == 0);

  rows = src.rows / block_size;
  cols = src.cols / block_size;
  create(rows * cols);

  for (int r = 0; r < new_rows; r++) {
    for (int c = 0; c < new_cols; c++) {
      int order_block = (r % block_size) * block_size + (c % block_size);
      int idx_mat = (r / block_size) * cols + (c / block_size);
      memories[order_block][idx_mat] = src.at<ushort>(r, c);
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

static void spread(Mat &src, Mat &dst, int kernel_size) {
  dst = Mat::zeros(src.size(), QUANTIZE_TYPE);
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          int u = i + dy;
          int v = j + dx;
          if (u < 0 || u >= src.rows || v < 0 || v >= src.cols ||
              !src.at<quantize_type>(u, v))
            continue;
          dst.at<quantize_type>(i, j) |= dst.at<quantize_type>(u, v);
        }
      }
    }
  }
}

#include "similariry_lut.i"
const int bit_mask[] = {15, 240, 3840, 61440};

static void computeResponseMaps(Mat &src, vector<Mat> &response_maps) {
  response_maps.resize(QUANTIZE_BASE);
  for (int i = 0; i < 16; i++)
    response_maps[i] = Mat::zeros(src.size(), CV_16U);

  static int bit_size = QUANTIZE_BASE / 4;
  static int lut_step = bit_size * QUANTIZE_BASE;

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (!src.at<quantize_type>(i, j))
        continue;

      uchar _bit_[bit_size];
      for (int k = 0; k < bit_size; k++)
        _bit_[k] = src.at<quantize_type>(i, j) & bit_mask[k];

      for (int ori = 0; ori < QUANTIZE_BASE; ori++) {
        uchar maxScore = 0;
        for (int k = 0; k < bit_size; k++)
          maxScore =
              max(maxScore, similarity_lut[ori * lut_step + 16 * k + _bit_[k]]);
        response_maps[ori].at<quantize_type>(i, j) = maxScore;
      }
    }
  }
}

void Detector::addSource(cv::Mat &src, cv::Mat mask,
                         const cv::String &memory_name) {
  auto named_memory = memories_map.find(memory_name);
  if (named_memory != memories_map.end()) {
    printf("该名称被占用! 是否重新写入源图像 [y/n]\n");
    char ch = getchar();
    if (ch == 'y' || ch == 'Y')
      ;
    else if (ch == 'n' || ch == 'N')
      return;
  }

  modality = modality->process(src, mask);
  vector<LinearMemory> memories_pyramid;

  for (int l = 0; l < pyramid_level; l++) {
    vector<LinearMemory> linear_memories;
    for (int i = 0; i < QUANTIZE_BASE; i++)
      linear_memories.push_back(LinearMemory(block_size));

    Mat quantized, spread_quantized;
    modality->quantize(quantized);
    spread(quantized, spread_quantized, 3);

    vector<Mat> response_maps;
    computeResponseMaps(spread_quantized, response_maps);

    for (int i = 0; i < QUANTIZE_BASE; i++)
      linear_memories[i].linearize(response_maps[i]);

    memories_pyramid.insert(memories_pyramid.end(), linear_memories.begin(),
                            linear_memories.end());

    if (l != pyramid_level - 1)
      modality->pyrDown();
  }

  memories_map.insert(make_pair(memory_name, memories_pyramid));
}

void Detector::addTemplate(cv::Mat &object, cv::Mat object_mask,
                          const Search &search, const cv::String &templ_name) {
  auto named_templ = templates_map.find(templ_name);
  if (named_templ != templates_map.end()) {
    printf("该名称被占用! 是否重新写入模板 [y/n]\n");
    char ch = getchar();
    if (ch == 'y' || ch == 'Y')
      ;
    else if (ch == 'n' || ch == 'N')
      return;
  }

  modality = modality->process(object, object_mask);

  vector<Ptr<ShapeTemplate> > templs;

  const Range &scale_range = search.scale;
  const Range &angle_range = search.angle;

  for (int l = 0; l < pyramid_level; l++) {
    ShapeTemplate origin_tmepl(l, 1.0f, 0.0f);
    modality->extractTemplate(origin_tmepl);

    for (float scale = scale_range.lower_bound;
         scale < scale_range.upper_bound + line2d_eps; scale += scale_range.step) {
      for (float angle = angle_range.lower_bound;
           angle < angle_range.upper_bound + line2d_eps; angle += angle_range.step) {
        templs.push_back(origin_tmepl.relocate(scale, angle));
      }
    }

    if (l != pyramid_level - 1)
      modality->pyrDown();
  }

  templates_map.insert(make_pair(templ_name, templs));
  addSearch(scale_range, angle_range, templ_name);
}

void Detector::addSearch(Range scale, Range angle,
                         const cv::String &search_name) {
  auto named_search = searches_map.find(search_name);
  if (named_search != searches_map.end()) {
    printf("该名称被占用! 是否重新写入搜索范围 [y/n]\n");
    char ch = getchar();
    if (ch == 'y' || ch == 'Y')
      ;
    else if (ch == 'n' || ch == 'N')
      return;
  }

  searches_map.insert(make_pair(search_name, Search(scale, angle)));
}

static void computeSimilarity(LinearMemory *response_map,
                              const ShapeTemplate &templ,
                              LinearMemory &similarity) {
  similarity.create(response_map[0].linear_size(), 0);
  similarity.cols = response_map[0].cols;
  similarity.rows = response_map[0].rows;

  int size = similarity.block_size * similarity.block_size;

  for (int i = 0; i < size; i++) {
    for (const auto &point : templ.features) {
      Point cur = Point(point.x + i % 4, point.y + i / 4);

      int mod_y = cur.y % 4 < 0 ? (cur.y % 4) + 4 : cur.y % 4;
      int mod_x = cur.x % 4 < 0 ? (cur.x % 4) + 4 : cur.x % 4;

      int offset = ((cur.y - mod_y) / 4) * similarity.cols + (cur.x - mod_x) / 4;

      for (int j = 0; j < similarity.linear_size(); j++) {
        similarity.at(i, j) +=
            response_map[point.label].at(mod_y * 4 + mod_x, j + offset);
      }
    }
  }
}

static void addLocalSimilarity(LinearMemory *response_map,
                              const ShapeTemplate &templ,
                              LinearMemory &similarity, int x, int y) {
  int n_rows = similarity.rows * 4;
  int n_cols = similarity.cols * 4;
  for (int i = y - 8; i < y + 8; y++) {
    for (int j = x - 8; j < x + 8; j++) {
      for (const auto &point : templ.features) {
        Point cur = Point(point.x + j, point.y + i);

        if (cur.y < 0 || cur.x < 0 || cur.y >= n_rows || cur.x >= n_cols)
          continue;

        similarity.linear_at(i, j) += response_map->linear_at(cur.y, cur.x);
      }
    }
  }
}

// Used to filter out weak matches
struct MatchPredicate {
  float threshold;
  MatchPredicate(float _threshold) : threshold(_threshold) {}
  bool operator()(const Match &m) { return m.similarity < threshold; }
};

void Detector::match(cv::Mat &src, cv::Mat &object, 
                     float score_threshold,
                     const Search &search,
                     cv::Mat src_mask, cv::Mat object_mask) {
  int area = sqrt(object.rows * object.cols);
  int try_level = 1;
  while (128 * (1 << try_level) < area)
    try_level++;
  pyramid_level = try_level;

  addSource(src, src_mask);
  addTemplate(object, object_mask, search);

  matchClass("default", "default", score_threshold);
}

void Detector::matchClass(const cv::String &match_name,
                          const cv::String &search_name, float score_threshold) {
  vector<Ptr<ShapeTemplate> > &vtp = templates_map[match_name];
  vector<LinearMemory> &vlm = memories_map[match_name];
  vector<Match> matches;
  int num_templates = vtp.size() / pyramid_level;

  for (int template_id = 0; template_id < num_templates; template_id++) {
    int match_level = pyramid_level - 1;
    Ptr<ShapeTemplate> templ = vtp[match_level * num_templates + template_id];
    LinearMemory *response_map_begin = &vlm[match_level * QUANTIZE_BASE];

    LinearMemory similarity(block_size);
    computeSimilarity(response_map_begin, *templ, similarity);

    int num_features = (*templ).features.size();
    int raw_threshold = static_cast<int>(4 * num_features + 
                        (score_threshold / 100.0f) * (4 * num_features) + 0.5f);

    vector<Match> candidates;
    for (int r = 0; r < similarity.rows; r++) {
      for (int c = 0; c < similarity.cols; c++) {
        int raw_score = similarity.linear_at(r, c);
        if (raw_score > raw_threshold) {
          float score = (raw_score * 100.0f) / (8 * num_features) + 0.5f;
          candidates.push_back(Match(c, r, score, match_name, template_id));
        }
      }
    }

    for (int l = match_level - 1; l >= 0; l--) {
      Ptr<ShapeTemplate> templ = vtp[match_level * num_templates + template_id];
      LinearMemory *response_map_begin = &vlm[match_level * QUANTIZE_BASE];

      LinearMemory local_similarity(block_size);
      local_similarity.create(response_map_begin->linear_size(), 0);
      local_similarity.rows = response_map_begin->rows;
      local_similarity.cols = response_map_begin->cols;
      for (int k = 0; k < (int)candidates.size(); k++) {
        Match &point = candidates[k];
        int x = point.x * 2;
        int y = point.y * 2;

        addLocalSimilarity(response_map_begin, *templ, local_similarity, x, y);

        int best_score = 0;
        Point best_match(-1, -1);
        for (int r = 0; r < local_similarity.rows; r++) {
          for (int l = 0; l < local_similarity.cols; l++) {
            int score = local_similarity.linear_at(r, l);
            if (score > best_score) {
              best_score = score;
              best_match = Point(l, r);
            }
          }
        }

        point.x = best_match.x;
        point.y = best_match.y;
        point.similarity = (best_score * 100.0f) / (8 * num_features);
      }

      // Filter out any matches that drop below the similarity threshold
      vector<Match>::iterator new_end = remove_if(
          candidates.begin(), candidates.end(), MatchPredicate(score_threshold));
      candidates.erase(new_end, candidates.end());
    }

    matches.insert(matches.end(), candidates.begin(), candidates.end());
  }

  matches_map.insert(make_pair(match_name, matches));
}

void Detector::detectBestMatch(vector<Vec6f> &points, 
                               vector<RotatedRect> &boxes,
                               const String &match_name) {
  vector<Ptr<ShapeTemplate> > &vtp = templates_map[match_name];
  vector<Match> &matches = matches_map[match_name];

  vector<bool> count(vtp.size() / pyramid_level, false);
  points.resize(matches.size());

  for (int i = 0; i < (int)points.size(); i++) {
    int temp_id = matches[i].template_id;
    if (!count[temp_id]) {
      boxes.push_back(vtp[temp_id]->box);
      count[temp_id] = true;
    }
    points[i][0] = matches[i].x;
    points[i][1] = matches[i].y;
    points[i][2] = vtp[temp_id]->scale;
    points[i][3] = vtp[temp_id]->angle;
    points[i][4] = matches[i].similarity;
  }
}
