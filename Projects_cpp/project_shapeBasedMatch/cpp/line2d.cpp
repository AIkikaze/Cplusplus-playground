#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;
double __time__relocate__ = 0.0;
double __time__produceroi__ = 0.0;

void __onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    Mat *image = static_cast<Mat *>(userdata);
    if (image != nullptr && !image->empty()) {
      if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
        auto pixel = image->at<float>(y, x);
        std::cout << "Pixel value at (" << x << ", " << y << "): " << pixel
                  << std::endl;
      }
    }
  }
}

vector<float> Detector::cos_table;

ImagePyramid::ImagePyramid() {
  pyramid_level = 0;
  pyramid = vector<Mat>();
}

ImagePyramid::ImagePyramid(const Mat &src, int py_level) {
  buildPyramid(src, py_level);
}

Mat &ImagePyramid::operator[](int index) {
  CV_Assert(index >= 0 && index < pyramid_level);
  return pyramid[index];
}

void ImagePyramid::buildPyramid(const Mat &src, int py_level) {
  pyramid_level = py_level;
  Mat curImg = src.clone();

  // 裁剪图像到合适尺寸
  int suit_size =
      (1 << pyramid_level) * 4;  // 4: 使得最高层图像长宽为 4 的倍数便于线性化
  int n_rows = (curImg.rows / suit_size + 1) * suit_size;
  int n_cols = (curImg.cols / suit_size + 1) * suit_size;
  copyMakeBorder(curImg, curImg, 0, n_rows - curImg.rows, 0,
                     n_cols - curImg.cols, BORDER_REPLICATE);

  // 初始化图像金字塔
  pyramid.push_back(curImg);

  // 构建图像金字塔
  for (int i = 0; i < pyramid_level - 1; i++) {
    Mat downsampledImg;
    pyrDown(pyramid[i], downsampledImg);
    pyramid.push_back(downsampledImg);
  }
}

shapeInfo_producer::shapeInfo_producer() {
  angle_range[0] = angle_range[1] = 0.0f;
  scale_range[0] = scale_range[1] = 1.0f;
  angle_step = 0.0f;
  scale_step = 0.0f;
  eps = line2d_eps;
}

shapeInfo_producer::shapeInfo_producer(const Mat &input_src,
                                       Mat input_mask, bool padding)
    : shapeInfo_producer() {
  CV_Assert(!input_src.empty());
  CV_Assert(input_mask.empty() || (input_src.size() == input_mask.size()));
  CV_Assert(input_mask.empty() || input_mask.type() == CV_8U);
  // 当前图像已经扩充边界
  if (padding) {
    src = input_src;
    if (input_mask.empty())
      mask = Mat(src.size(), CV_8U, Scalar(255));
    else
      mask = input_mask;
  } else {  // 当前图像未扩充 0 边界
    // 图像在旋转和缩放过程中有效像素到图像中心的最远距离
    int border_max = 1 + (int)sqrt(input_src.rows * input_src.rows +
                                   input_src.cols * input_src.cols);
    // 扩充边界
    copyMakeBorder(input_src, src, border_max - input_src.rows / 2,
                       border_max - input_src.rows / 2,
                       border_max - input_src.cols / 2,
                       border_max - input_src.cols / 2, BORDER_REPLICATE);

    if (input_mask.empty())
      mask = Mat(src.size(), CV_8U, Scalar(255));
    else  // 扩充掩图边界
      copyMakeBorder(input_mask, mask, border_max - input_mask.rows / 2,
                         border_max - input_mask.rows / 2,
                         border_max - input_mask.cols / 2,
                         border_max - input_mask.cols / 2, BORDER_CONSTANT);
  }
}

void shapeInfo_producer::save_config(const shapeInfo_producer &sip,
                                     string path) {
  FileStorage fs(path, FileStorage::WRITE);

  // 储存公有成员参数
  fs << "params"
     << "{";
  fs << "angle_range"
     << "[:" << sip.angle_range[0] << sip.angle_range[1] << "]";
  fs << "scale_range"
     << "[:" << sip.scale_range[0] << sip.scale_range[1] << "]";
  fs << "angle_step" << sip.angle_step;
  fs << "scale_step" << sip.scale_step;
  fs << "eps" << sip.eps;
  fs << "}";

  // 储存日志列表
  fs << "infos"
     << "[";
  for (const auto &info : sip.Infos_constptr()) {
    fs << "{";
    fs << "angle" << info.angle;
    fs << "scale" << info.scale;
    fs << "}";
  }
  fs << "]";

  fs.release();
}

Ptr<shapeInfo_producer> shapeInfo_producer::load_config(
    const Mat &input_src, Mat input_mask, bool padding, string path) {
  FileStorage fs(path, FileStorage::READ);

  // 用 Ptr 管理并初始化 shapeInfo_producer 对象
  Ptr<shapeInfo_producer> sip =
      makePtr<shapeInfo_producer>(input_src, input_mask, padding);

  // 读取公有成员参数
  FileNode paramsNode = fs["params"];
  sip->angle_range[0] = (float)paramsNode["angle_range"][0];
  sip->angle_range[1] = (float)paramsNode["angle_range"][1];
  sip->scale_range[0] = (float)paramsNode["scale_range"][0];
  sip->scale_range[1] = (float)paramsNode["scale_range"][1];
  sip->angle_step = (float)paramsNode["angle_step"];
  sip->scale_step = (float)paramsNode["scale_step"];
  sip->eps = (float)paramsNode["eps"];

  // 读取日志列表
  vector<Info> &infos = sip->Infos_ptr();
  for (const auto &info : fs["infos"]) {
    infos.push_back(Info((float)info["angle"], (float)info["scale"]));
  }

  fs.release();

  return sip;
}

void shapeInfo_producer::produce_infos() {
  if (!Infos.empty()) Infos.clear();
  CV_Assert(scale_range[0] < scale_range[1] + eps);
  CV_Assert(angle_range[0] < angle_range[1] + eps);
  if (scale_step < eps) scale_step = 2 * eps;
  if (angle_step < eps) angle_step = 2 * eps;
  for (float scale = scale_range[0]; scale <= scale_range[1] + eps;
       scale += scale_step)
    for (float angle = angle_range[0]; angle <= angle_range[1] + eps;
         angle += angle_step) {
      Infos.push_back(Info(angle, scale));
    }
}

Mat shapeInfo_producer::affineTrans(const Mat &src, float angle,
                                        float scale) {
  Mat dst;
  Point2f center(cvFloor(src.cols / 2.0f), cvFloor(src.rows / 2.0f));
  Mat rotate_mat = getRotationMatrix2D(center, angle, scale);
  warpAffine(src, dst, rotate_mat, src.size(), INTER_LINEAR,
                 BORDER_REPLICATE);
  return dst;
}

Mat shapeInfo_producer::src_of(Info info) {
  return affineTrans(src, info.angle, info.scale);
}

Mat shapeInfo_producer::mask_of(Info info) {
  return affineTrans(mask, info.angle, info.scale);
}

const vector<shapeInfo_producer::Info> &shapeInfo_producer::Infos_constptr()
    const {
  return Infos;
}

int line2d::angle2ori(const float &angle) {
  float angle_mod = angle > 180 ? angle - 180 : angle;
  int orientation = 0;
  for (int i = 0; i < 16; i++) {
    if (angle_mod <= (float)((i + 1) * 180.0f / 16.0f)) {
      orientation = i;
      break;
    }
  }
  return orientation;
}

float line2d::ori2angle(const int &orientation) {
  return 5.625f + 11.25f * orientation;
}

/// Template

Template::Template() {
  nms_kernel_size = defaultParams.nms_kernel_size;
  magnitude_threshold = defaultParams.magnitude_threshold;
  num_features = defaultParams.num_features;
  template_created = defaultParams.template_created;
}

Template::Template(TemplateParams params, bool isDefault) {
  nms_kernel_size = params.nms_kernel_size;
  magnitude_threshold = params.magnitude_threshold;
  num_features = params.num_features;
  template_created = params.template_created;
  if (isDefault) defaultParams = params;
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


bool Template::createTemplate(const Mat &src, Template &tp,
                              int nms_kernel_size) {
  Mat magnitude, sobel_dx, sobel_dy;
  sobelMagnitude(src, magnitude, sobel_dx, sobel_dy);

  Mat sobel_ag;
  phase(sobel_dx, sobel_dy, sobel_ag, true);

  vector<Feature> candidates;
  for (int r = 0; r < magnitude.rows; r++) {
    for (int l = 0; l < magnitude.cols; l++) {
      const float &angle_at_rl = sobel_ag.at<ushort>(r, l);
      const float &magnitude_at_rl = magnitude.at<float>(r, l);
      if (angle_at_rl > 0 && magnitude_at_rl > tp.magnitude_threshold) {
        candidates.push_back(
            Feature(l, r, angle_at_rl, magnitude_at_rl));
      }
    }
  }

  if (candidates.size() < tp.num_features)
    return false;

  stable_sort(candidates.begin(), candidates.end());

  float distance = static_cast<float>(candidates.size() / tp.num_features + 1);
  tp.selectFeatures_from(candidates, distance);
  return true;
}

Ptr<Template> Template::createPtr_from(const Mat &src,
                                           TemplateParams params) {
  Ptr<Template> tp = makePtr<Template>(params);

  for (int i = 0; i < 100; i++) {
    createTemplate(src, *tp, tp->nms_kernel_size);
    if (tp->features.size() >= tp->num_features)
      break;
    else {
      if (tp->nms_kernel_size > 3)
        tp->nms_kernel_size -= 2;
    }
  }
  if (tp->features.size() >= tp->num_features)
    tp->template_created = true;
  else {
    cerr << "无法找到的足够的特征点！" << endl;
    tp->features.clear();
  }

  return tp;
}

void line2d::Template::create_from(const Mat &src) {
  for (int i = 0; i < 100; i++) {
    createTemplate(src, *this, nms_kernel_size);
    if (features.size() >= num_features)
      break;
    else {
      if (nms_kernel_size > 3)
        nms_kernel_size -= 2;
    }
  }
  if (features.size() >= num_features)
    template_created = true;
  else {
    cerr << "无法找到的足够的特征点！" << endl;
    features.clear();
  }
}

void Template::selectFeatures_from(vector<Feature> &candidates, float distance) {
  CV_Assert(distance > 2.0f);
  CV_Assert(!candidates.empty());

  features.clear();
  float distance_sq = distance * distance;
  int i = 0;

  while (features.size() < num_features) {
    const Feature &c = candidates[i];
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

vector<Template::Feature> Template::relocate_by(shapeInfo_producer::Info info) {
  Timer time;
  float theta = -info.angle / 180.0f * CV_PI;
  vector<Feature> new_prograds;
  for (const auto &pg : features) {
    Point2f newPoint;
    newPoint.x = info.scale * (cos(theta) * pg.x - sin(theta) * pg.y);
    newPoint.y = info.scale * (sin(theta) * pg.x + cos(theta) * pg.y);
    new_prograds.push_back(Feature(newPoint.x, newPoint.y,
                                   fmod(pg.angle - info.angle + 360.0f, 360.0f),
                                   pg.score));
  }
  __time__relocate__ += time.elapsed();
  return new_prograds;
}

void Template::show_in(Mat &background, Point center,
                       shapeInfo_producer::Info info) {
  if (background.type() == CV_8U)
    cvtColor(background, background, COLOR_GRAY2BGR);

  Point leftup(background.cols, background.rows);
  Point rightdown(0, 0);
  vector<Feature> new_prograds = relocate_by(info);
  for (const auto &pg : new_prograds) {
    Point cur = center + Point(pg.x, pg.y);
    circle(background, cur, 1, Scalar(0, 0, 255), -1);
    leftup.x = min(leftup.x, cur.x);
    leftup.y = min(leftup.y, cur.y);
    rightdown.x = max(rightdown.x, cur.x);
    rightdown.y = max(rightdown.y, cur.y);
  }
  leftup.x = max(leftup.x - 5, 0);
  leftup.y = max(leftup.y - 5, 0);
  rightdown.x = min(rightdown.x + 5, background.cols - 1);
  rightdown.y = min(rightdown.y + 5, background.rows - 1);
  rectangle(background, Rect(leftup, rightdown), Scalar(0, 255, 0),
                1);
  circle(background, center, 1, Scalar(0, 255, 0), -1);
}

Detector::Detector() {
  pyramid_level = 2;
  init_costable();
}

void Detector::quantize(const Mat &edges, const Mat &angles,
                        Mat &ori_bit, int kernel_size, float magnitude_threshold) {
  ori_bit = Mat::zeros(angles.size(), CV_16U);
  // Mat ori_mat = Mat::ones(angles.size(), CV_8U);
  // ori_mat *= 125;
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<float>(i, j) <= magnitude_threshold) continue;
      int max_count = 0;
      int orientation = 0;
      int ori_count[16] = {0};
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          if (i + dy < 0 || i + dy >= edges.rows || j + dx < 0 ||
              j + dx >= edges.cols)
            continue;
          if (edges.at<float>(i + dy, j + dx) > magnitude_threshold) {
            int _ori = angle2ori(angles.at<float>(i + dy, j + dx));
            ori_count[_ori]++;
            if (ori_count[_ori] > max_count) {
              max_count = ori_count[_ori];
              orientation = _ori;
            }
          }
        }
      }
      // if (ori_count[orientation] > (kernel_size * kernel_size) / 2 + 1) {
      ori_bit.at<ushort>(i, j) = 1 << orientation;
      // ori_mat.at<uchar>(i, j)  = orientation * 16;
      // }
    }
  }
  // namedWindow("ori_mat", WINDOW_NORMAL);
  // imshow("ori_mat", ori_mat);
  // waitKey();
}

void Detector::spread(Mat &ori_bit, Mat &spread_ori, int kernel_size) {
  spread_ori = Mat::zeros(ori_bit.size(), CV_16U);
  for (int i = 0; i < ori_bit.rows; i++) {
    for (int j = 0; j < ori_bit.cols; j++) {
      // cout << "(debug)" << endl;
      // cout << "curPoint:" << i << "," << j << endl;
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          int u = i + dy;
          int v = j + dx;
          if (u < 0 || u >= ori_bit.rows || v < 0 || v >= ori_bit.cols ||
              !ori_bit.at<ushort>(u, v))
            continue;
          // cout << "point in kernel:" << u << "," << v << endl;
          // cout << "oir_bit:" << bitset<16>(ori_bit.at<ushort>(u, v)) << endl;
          spread_ori.at<ushort>(i, j) |= ori_bit.at<ushort>(u, v);
          // cout << "spread_ori:" << bitset<16>(spread_ori.at<ushort>(i, j)) <<
          // endl; cin.get();
        }
      }
    }
  }
}

void Detector::computeResponseMaps(Mat &spread_ori,
                                   vector<Mat> &response_maps) {
  response_maps.resize(16);
  for (int i = 0; i < 16; i++)
    response_maps[i] = Mat::zeros(spread_ori.size(), CV_32F);

  int maxValue = numeric_limits<ushort>::max();
  for (int i = 0; i < spread_ori.rows; i++) {
    for (int j = 0; j < spread_ori.cols; j++) {
      if (!spread_ori.at<ushort>(i, j)) continue;
      for (int orientation = 0; orientation < 16; orientation++) {
        // cout << "(debug)" << endl;
        response_maps[orientation].at<float>(i, j) =
            cos_table[orientation * maxValue +
                      (int)spread_ori.at<ushort>(i, j)];
        // for (int k = 0; k < 16; k++) {
        //   cout << "cos_table at [" << k << "," << i << "," << j
        //        << "]:" << cos_table[k * maxValue +
        //               (int)spread_ori.at<ushort>(i, j)] << endl;
        // }
        // for (int k = 0; k < 16; k++) {
        //   cout << "response_map at [" << k << "," << i << "," << j
        //        << "]:" << response_maps[k].at<float>(i, j) << endl;
        // }
        // cin.get();
      }
      // cin.get();
    }
  }
}

void Detector::para_computeSimilarityMap(
    vector<LinearMemories> &memories, const vector<Template::Feature> &features,
    LinearMemories &similarity, int start, int end) {
  // 并行计算
  for (int i = start; i < end; i++) {
    for (const auto &point : features) {
      Point cur = Point(point.x + i % 4, point.y + i / 4);

      int mod_y = cur.y % 4 < 0 ? (cur.y % 4) + 4 : cur.y % 4;
      int mod_x = cur.x % 4 < 0 ? (cur.x % 4) + 4 : cur.x % 4;

      int offset =
          ((cur.y - mod_y) / 4) * similarity.cols + (cur.x - mod_x) / 4;

      for (int j = 0; j < similarity.linear_size(); j++) {
        similarity.at(i, j) +=
            memories[point.label].at(mod_y * 4 + mod_x, j + offset);
      }
    }
  }
}

void Detector::computeSimilarityMap(vector<LinearMemories> &memories,
                                    const vector<Template::Feature> &features,
                                    LinearMemories &similarity) {
  // similarity[i][j]
  // i -> order in kernel ; j -> index in linear vector
  similarity.create(16, memories[0].linear_size(), 0.0f);
  similarity.rows = memories[0].rows;
  similarity.cols = memories[0].cols;

  const int numThreads = thread::hardware_concurrency();
  const int workloadPerThread = (16 + numThreads - 1) / numThreads;

  vector<thread> threads;
  // atomic<int> currentRow(0);

  // 启动多个线程并行计算
  for (int t = 0; t < numThreads; t++) {
    int start = t * workloadPerThread;
    int end = min(start + workloadPerThread, 16);

    threads.emplace_back([&memories, &features, &similarity, start, end] {
      para_computeSimilarityMap(memories, features, similarity, start, end);
    });

    if (end == 16) break;
  }

  // 等待所有线程完成
  for (auto &thread : threads) {
    thread.join();
  }

  // 转化为 100 分制
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < memories[0].linear_size(); j++) {
      similarity.at(i, j) =
          similarity.at(i, j) / (float)features.size() * 100.0f;
    }
  }
}

void Detector::localSimilarityMap(vector<LinearMemories> &memories,
                                  const vector<Template::Feature> &features,
                                  Mat &similarity_map,
                                  vector<Rect> &rois) {
  CV_Assert(!rois.empty());
  int n_rows = memories[0].rows * 4;
  int n_cols = memories[0].cols * 4;

  similarity_map = Mat::zeros(n_rows, n_cols, CV_32F);

  for (size_t k = 0; k < rois.size(); k++) {
    for (int i = rois[k].y; i <= rois[k].y + rois[k].height; i++) {
      for (int j = rois[k].x; j <= rois[k].x + rois[k].width; j++) {
        for (const auto &point : features) {
          Point cur = Point(point.x + j, point.y + i);

          if (cur.y < 0 || cur.x < 0 || cur.y >= n_rows || cur.x >= n_cols)
            continue;

          // 计算坐标在线性存储器中的当前位置
          int position_cur = cur.y / 4 * memories[0].cols + cur.x / 4;
          // 计算坐标在 TxT 分块中的顺序索引
          int order_kernel = (cur.y % 4) * 4 + cur.x % 4;
          // 计算相似度矩阵
          similarity_map.at<float>(i, j) +=
              memories[point.label].at(order_kernel, position_cur);
        }
      }
    }
  }

  // 转化为 100 分制
  similarity_map = similarity_map / (float)features.size() * 100.0f;
}

void Detector::linearize(std::vector<Mat> &response_maps,
                         vector<LinearMemories> &linearized_memories) {
  // 计算分块后矩阵的行数, 列数
  int n_rows = response_maps[0].rows / 4;
  int n_cols = response_maps[0].cols / 4;

  // 初始化 linearized_memories, 16 -> 16 个量化方向
  linearized_memories.resize(16);
  for (int i = 0; i < 16; i++) {
    linearized_memories[i].create(16,
                                  n_rows * n_cols);  // 16 -> 线性化单元块的大小
    linearized_memories[i].rows = n_rows;  // 记录行数 (便于后续还原)
    linearized_memories[i].cols = n_cols;  // 记录列数 (便于后续还原)
  }

  // cout << "(debug)" << endl;
  // cout << "分块矩阵大小:" << n_rows << "x" << n_cols << endl;
  for (int orientation = 0; orientation < 16; orientation++) {
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        for (int di = 0; di < 4; di++) {
          for (int dj = 0; dj < 4; dj++) {
            // if (abs(response_maps[orientation].at<float>(4 * i + di, 4 * j +
            // dj)) < line2d_eps) continue; cout << "分块坐标:" << i << "," << j
            // << endl; cout << "单像素坐标:" << 4*i + di << "," << 4*j + dj <<
            // endl; cout << "order in kernel:" << di * 4 + dj << endl; cout <<
            // "index in linear vector:" << i * n_cols + j << endl; cout <<
            // "response_maps:" << response_maps[orientation].at<float>(4 * i +
            // di, 4 * j + dj) << endl; cout << "linearized_memories:" <<
            // linearized_memories[orientation].at(di * 4 + dj,
            //                                     i * n_cols + j) << endl;
            int order_in_kernel = di * 4 + dj;
            linearized_memories[orientation].at(order_in_kernel,
                                                i * n_cols + j) =
                response_maps[orientation].at<float>(4 * i + di, 4 * j + dj);
            // cout << "linearized_memories:" <<
            // linearized_memories[orientation].at(di * 4 + dj,
            //                                     i * n_cols + j) << endl;
            // cin.get();
          }
        }
      }
    }
  }
}

void Detector::unlinearize(LinearMemories &similarity,
                           Mat &similarity_map) {
  similarity_map =
      Mat::zeros(similarity.rows * 4, similarity.cols * 4, CV_32F);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < similarity.linear_size(); j++) {
      int u = (j / similarity.cols) * 4 + i / 4;
      int v = (j % similarity.cols) * 4 + i % 4;
      similarity_map.at<float>(u, v) = similarity.at(i, j);
    }
  }
}

void Detector::produceRoi(Mat &similarity_map,
                          std::vector<Rect> &roi_list, int lower_score) {
  Timer time;
  Mat binary_mat;
  Mat labels, stats, centroids;

  threshold(similarity_map, binary_mat, lower_score, 255,
                THRESH_BINARY);
  if (binary_mat.type() != CV_8U) binary_mat.convertTo(binary_mat, CV_8U);

  int numLabels =
      connectedComponentsWithStats(binary_mat, labels, stats, centroids);

  roi_list.clear();
  for (int i = 1; i < numLabels; i++) {
    Rect roi(stats.at<int>(i, CC_STAT_LEFT),
                 stats.at<int>(i, CC_STAT_TOP),
                 stats.at<int>(i, CC_STAT_WIDTH),
                 stats.at<int>(i, CC_STAT_HEIGHT));
    roi_list.push_back(roi);
  }
  __time__produceroi__ += time.elapsed();
}

void Detector::addSourceImage(const Mat &src, int pyramid_level, Mat mask,
                              const String &memories_id) {
  ImagePyramid src_pyramid;
  Mat target_src;
  if (!mask.empty())
    src.copyTo(src, mask);
  else
    target_src = src.clone();
  
  src_pyramid.buildPyramid(target_src, pyramid_level);

  memory_pyramid mp;
  mp.reserve(pyramid_level);
  for (int i = 0; i < pyramid_level; i++) {
    Mat magnitude, sobel_dx, sobel_dy;
    sobelMagnitude(src_pyramid[i], magnitude, sobel_dx, sobel_dy);

    Mat sobel_ag;
    phase(sobel_dx, sobel_dy, sobel_ag, true);

    Mat ori_bit;
    quantize(magnitude, sobel_ag, ori_bit, 3, 0.2f);

    Mat spread_ori;
    spread(ori_bit, spread_ori, 3);

    vector<Mat> response_maps;
    computeResponseMaps(spread_ori, response_maps);

    linearize(response_maps, mp[i]);
  }
  memories_map.insert(std::make_pair(memories_id, mp));
}

void Detector::addTemplate(const Mat &temp_src, int pyramid_level,
                          Template::TemplateParams params, Mat mask,
                          shapeInfo_producer *sip,
                          const String &templates_id) {
    ImagePyramid temp_pyramid;
    Mat target_temp;
    if (!mask.empty())
      temp_src.copyTo(target_temp, mask);
    else
      target_temp = temp_src.clone();

    temp_pyramid.buildPyramid(target_temp, pyramid_level);

    vector<template_pyramid> vtp;

    vector<shapeInfo_producer::Info> infos;
    if (sip == nullptr || sip->Infos_constptr().empty()) {
      infos.push_back(shapeInfo_producer::Info());
    }
    else 
      infos = sip->Infos_constptr();

    for (const auto &info : infos) {
      template_pyramid tp;
      for (int i = 0; i < pyramid_level; i++) {
        Mat template_mat = sip->affineTrans(temp_pyramid[i], info.scale, info.angle);
        Ptr<Template> templ = Template::createPtr_from(template_mat, params);
        Mat tempImage = template_mat.clone();
        templ->show_in(tempImage,
                    Point(temp_pyramid[i].cols / 2, temp_pyramid[i].rows /
                    2));
        namedWindow("tempImage", WINDOW_NORMAL);
        imshow("tempImage", tempImage);
        waitKey();
        destroyWindow("tempImage");

        if (templ->iscreated())
          tp.push_back(templ);
        else
        cerr << "模板创建失败" << endl;

      // 更新下一层构造模板所需的参数
      if (params.nms_kernel_size > 3) params.nms_kernel_size -= 2;
      if (params.num_features > 40)
        params.num_features = params.num_features >> 1;
    }
    vtp.push_back(tp);
  }

  templates_map.insert(make_pair(templates_id, vtp));
}

void Detector::match(const Mat &sourceImage,
                     shapeInfo_producer *sip, int lower_score,
                     Template::TemplateParams params,
                     Mat mask_src) {
  int min_CR = min(sourceImage.rows, sourceImage.cols);
  int try_level = 1;
  while (128 * (1 << try_level) < min_CR) try_level++;
  pyramid_level = try_level;

  Timer _time;
  Timer __time;

  addSourceImage(sourceImage, pyramid_level, mask_src);
  __time.out("目标图片梯度响应初始化!");

  addTemplate(sip->src_of(), pyramid_level, params, sip->mask_of());
  __time.out("创建模板!");

  _time.out("__初始化完毕!__");

  double time1 = 0.0;
  double time2 = 0.0;

  matchClass("default");
  
  _time.out("__模板匹配计算完毕!__");
  cout << "全局计算时间: " << time1 << "s" << endl;
  cout << "局部计算时间: " << time2 << "s" << endl;
}

void line2d::Detector::matchClass(const cv::String &match_id) {
  memory_pyramid mp = memories_map[match_id];
  vector<template_pyramid> vtp = templates_map[match_id];
  matches points = matches_map[match_id];
  points.clear();

  for (int template_id = 0; template_id < (int)vtp.size(); template_id++) {
    const template_pyramid &tp = vtp[template_id];

    const vector<LinearMemories> &highest_lm = mp.back();

    int match_level = pyramid_level - 1;


  }
}

void Detector::draw(cv::Mat background, const cv::String &match_id, int template_id) {
  if (matches_map[match_id].empty()) {
    cerr << "没有找到匹配点!" << endl;
    return;
  }
  template_pyramid tp = templates_map[match_id][template_id];
  for (const auto &point : matches_map[match_id]) {
    tp[0]->show_in(background, Point(point.x, point.y), tp[0]->);
    Vec3b randColor;
    randColor[0] = rand() % 155 + 100;
    randColor[1] = rand() % 155 + 100;
    randColor[2] = rand() % 155 + 100;
    putText(background, to_string(int(round(point.similarity))),
                Point(point.x - 10, point.y - 3),
                FONT_HERSHEY_PLAIN, 2, randColor);
  }
  namedWindow("matchImage", WINDOW_NORMAL);
  imshow("matchImage", background);
  waitKey();
  destroyWindow("matchImage");
}

/// @todo 转为离线存储计算
void Detector::init_costable() {
  int maxValue = std::numeric_limits<ushort>::max();
  cos_table.resize(16 * maxValue);
  float maxCos;
  // Timer time;
  for (int i = 0; i < 16; i++) {
    // time.out(to_string(i+1)+"轮循环 65535 * 16 计算开始");
    for (int bit = 1; bit <= maxValue; bit++) {
      maxCos = -2.0f;  // < -1 = min_theta cos theta
      for (int k = 0; k < 16; k++) {
        if (bit & (1 << k)) {
          maxCos = maxCos < cos(abs(i - k) * _degree_(11.25f))
                       ? cos(abs(i - k) * _degree_(11.25f))
                       : maxCos;
        }
      }
      cos_table[i * maxValue + bit] = maxCos;
    }
    // time.out(to_string(i+1)+"轮循环 65535 * 16 计算结束");
    // cin.get();
  }
}