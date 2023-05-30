#include "line2d.hpp"
using namespace std;
using namespace cv;

ImagePyramid::ImagePyramid() {
  levels = 0;
  pyramid = vector<Mat>();
}

ImagePyramid::ImagePyramid(const Mat &src, int level_size) {
  levels = level_size;
  buildPyramid(src, levels);
}

Mat &ImagePyramid::operator[](int index) {
  CV_Assert(index >= 0 && index < levels);
  return pyramid[index];
}

void ImagePyramid::buildPyramid(const Mat &src, int levels) {
  Mat curImg = src.clone();

  // 裁剪图像到合适尺寸
  int suit_size =
      (1 << levels) * 4;  // 4: 使得最高层图像长宽为 4 的倍数便于线性化
  int n_rows = (curImg.rows / suit_size + 1) * suit_size;
  int n_cols = (curImg.cols / suit_size + 1) * suit_size;
  copyMakeBorder(curImg, curImg, 0, n_rows - curImg.rows, 0,
                 n_cols - curImg.cols, BORDER_REPLICATE);

  // 初始化图像金字塔
  pyramid.push_back(curImg);

  // 构建图像金字塔
  for (int i = 0; i < levels - 1; i++) {
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

shapeInfo_producer::shapeInfo_producer(const Mat &input_src, Mat input_mask,
                                       bool padding)
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

Ptr<shapeInfo_producer> shapeInfo_producer::load_config(const Mat &input_src,
                                                        Mat input_mask,
                                                        bool padding,
                                                        string path) {
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

Mat shapeInfo_producer::affineTrans(const Mat &src, float angle, float scale) {
  Mat dst;
  Point2f center(cvFloor(src.cols / 2.0f), cvFloor(src.rows / 2.0f));
  Mat rotate_mat = getRotationMatrix2D(center, angle, scale);
  warpAffine(src, dst, rotate_mat, src.size(), INTER_LINEAR, BORDER_REPLICATE);
  return dst;
}

Mat shapeInfo_producer::src_of(const Info &info) {
  return affineTrans(src, info.angle, info.scale);
}

Mat shapeInfo_producer::mask_of(const Info &info) {
  return affineTrans(mask, info.angle, info.scale);
}

const vector<shapeInfo_producer::Info> &shapeInfo_producer::Infos_constptr()
    const {
  return Infos;
}

int angle2ori(const float &angle) {
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

float ori2angle(const int &orientation) {
  return 180.0f / 32.0f + (180.0f / 16.0f) * orientation;
}

Template::Template() {
  TemplateParams defaultParams;
  nms_kernel_size = defaultParams.nms_kernel_size;
  scatter_distance = defaultParams.scatter_distance;
  grad_norm = defaultParams.grad_norm;
  num_features = defaultParams.num_features;
  template_created = defaultParams.template_crated;
}

Template::Template(TemplateParams params) {
  nms_kernel_size = params.nms_kernel_size;
  scatter_distance = params.scatter_distance;
  grad_norm = params.grad_norm;
  num_features = params.num_features;
  template_created = params.template_crated;
}

void Template::getOriMat(const cv::Mat &src, Mat &edges, Mat &angles) {
  CV_Assert(!src.empty());
  CV_Assert(src.channels() == 1 || src.channels() == 3);

  /// @variables:
  vector<Mat> imgs(3);
  split(src, imgs);  // 拆分成三个通道
  Mat gx = Mat::zeros(src.size(), CV_32F);
  Mat gy = Mat::zeros(src.size(), CV_32F);
  edges = Mat::zeros(src.size(), CV_32F);
  angles = Mat::zeros(src.size(), CV_32F);
  float max_grad_norm = 0.0f;

  // 计算梯度模长和方向角
  if (src.channels() == 1) {  // src 为单通道图像
    GaussianBlur(imgs[0], imgs[0], Size(5, 5), 1.5);
    Sobel(imgs[0], gx, CV_32F, 1, 0, 3);
    Sobel(imgs[0], gy, CV_32F, 0, 1, 3);
    for (int i = 0; i < edges.rows; i++) {
      for (int j = 0; j < edges.cols; j++) {
        edges.at<float>(i, j) = sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                                     gy.at<float>(i, j) * gy.at<float>(i, j));
        angles.at<float>(i, j) =
            fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));
        if (max_grad_norm < edges.at<float>(i, j))
          max_grad_norm = edges.at<float>(i, j);
      }
    }
  } else {  // src 为 3 通道图像
    vector<Mat> cedges(3, Mat::zeros(src.size(), CV_32F));
    vector<Mat> cangles(3, Mat::zeros(src.size(), CV_32F));
    for (int k = 0; k < 3; k++) {
      // GaussianBlur(imgs[k], imgs[k], Size(5, 5), 1.5);
      Scharr(imgs[k], gx, CV_32F, 1, 0);
      Scharr(imgs[k], gy, CV_32F, 0, 1);
      for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
          cedges[k].at<float>(i, j) =
              sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                   gy.at<float>(i, j) * gy.at<float>(i, j));
          cangles[k].at<float>(i, j) =
              fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));
        }
      }
    }
    // 取 3 通道中最大梯度模长对应的方向为方向角方向 Ori(O, r)
    for (int i = 0; i < edges.rows; i++) {
      for (int j = 0; j < edges.cols; j++) {
        float maxEdge = 0.0f;
        int index = 0;
        for (int k = 0; k < 3; k++) {
          if (cedges[k].at<float>(i, j) > maxEdge) {
            maxEdge = cedges[k].at<float>(i, j);
            index = k;
          }
        }
        if (max_grad_norm < maxEdge) max_grad_norm = maxEdge;
        edges.at<float>(i, j) = maxEdge;
        angles.at<float>(i, j) = cangles[index].at<float>(i, j);
      }
    }
  }

  // 归一化梯度模长矩阵
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      edges.at<float>(i, j) /= max_grad_norm;
    }
  }
}

void Template::createTemplate(const cv::Mat &src, Template &tp,
                              int nms_kernel_size, float scatter_distance) {
  Mat edges, angles;

  getOriMat(src, edges, angles);
  // Mat edgesImage;
  // normalize(edges, edgesImage, 0, 255, NORM_MINMAX, CV_8U);
  // imshow("edges", edgesImage);
  // waitKey();

  tp.selectFeatures_from(edges, angles, nms_kernel_size);

  tp.scatter(scatter_distance);
}

cv::Ptr<Template> Template::createPtr_from(const cv::Mat &src,
                                           TemplateParams params) {
  Ptr<Template> tp = makePtr<Template>(params);

  for (int i = 0; i < 100; i++) {
    createTemplate(src, *tp, tp->nms_kernel_size, tp->scatter_distance);
    if (tp->prograds.size() >= tp->num_features)
      break;
    else {
      if (tp->nms_kernel_size > 3)
        tp->nms_kernel_size -= 2;
      else {
        tp->scatter_distance +=
            (float)tp->nms_kernel_size * tp->num_features / tp->prograds.size();
      }
    }
  }
  if (tp->prograds.size() >= tp->num_features)
    tp->template_created = true;
  else {
    cerr << "无法找到的足够的特征点！" << endl;
    tp->prograds.clear();
  }

  return tp;
}

void Template::selectFeatures_from(const cv::Mat &_edges,
                                   const cv::Mat &_angles,
                                   int nms_kernel_size) {
  Mat edges = _edges.clone();
  Mat angles = Mat::zeros(_angles.size(), CV_8U);

  // 按方向角度进行非极大值抑制
  for (int i = 0; i < angles.rows; i++) {
    for (int j = 0; j < angles.cols; j++) {
      // 取方向角 0 ~ 360
      double angle = _angles.at<float>(i, j);

      // 方向角度量化
      if ((angle > 0 && angle < 22.5) || (angle > 157.5 && angle < 202.5) ||
          (angle > 337.5 && angle < 360))
        angles.at<uchar>(i, j) = 0;
      else if ((angle > 22.5 && angle < 67.5) ||
               (angle > 202.5 && angle < 247.5))
        angles.at<uchar>(i, j) = 45;
      else if ((angle > 67.5 && angle < 112.5) ||
               (angle > 247.5 && angle < 292.5))
        angles.at<uchar>(i, j) = 90;
      else if ((angle > 112.5 && angle < 157.5) ||
               (angle > 292.5 && angle < 337.5))
        angles.at<uchar>(i, j) = 135;
      else
        angles.at<uchar>(i, j) = 0;
    }
  }

  // 非极大值抑制
  Range rows(nms_kernel_size / 2, nms_kernel_size / 2 + edges.rows);
  Range cols(nms_kernel_size / 2, nms_kernel_size / 2 + edges.cols);
  copyMakeBorder(edges, edges, nms_kernel_size / 2, nms_kernel_size / 2,
                 nms_kernel_size / 2, nms_kernel_size / 2, BORDER_CONSTANT,
                 Scalar(0));
  for (int i = rows.start; i < rows.end; i++) {
    for (int j = cols.start; j < cols.end; j++) {
      float prevPixel, nextPixel;
      switch ((int)angles.at<uchar>(i - rows.start, j - cols.start)) {
        case 0:
          prevPixel = edges.at<float>(i, j - 1);
          nextPixel = edges.at<float>(i, j + 1);
          break;
        case 45:
          prevPixel = edges.at<float>(i - 1, j - 1);
          nextPixel = edges.at<float>(i + 1, j + 1);
          break;
        case 90:
          prevPixel = edges.at<float>(i - 1, j);
          nextPixel = edges.at<float>(i + 1, j);
          break;
        case 135:
          prevPixel = edges.at<float>(i + 1, j - 1);
          nextPixel = edges.at<float>(i - 1, j + 1);
          break;
      }

      // 当前点的梯度与梯度方向上临近点的梯度大小
      if (edges.at<float>(i, j) < prevPixel ||
          edges.at<float>(i, j) < nextPixel)
        edges.at<float>(i, j) = 0.0f;
    }
  }
  // Mat edgesImage;
  // normalize(edges, edgesImage, 0, 255, NORM_MINMAX, CV_8U);
  // namedWindow("edges after nms");
  // imshow("edges after nms", edges);
  // waitKey();
  // namedWindow("angles after quantized");
  // imshow("angles after quantized", angles);
  // waitKey();

  // 计算领域中出现次数最多的量化特征方向
  Mat orientation_map = Mat::zeros(_angles.size(), CV_8U);
  for (int i = rows.start; i < rows.end; i++) {
    for (int j = cols.start; j < cols.end; j++) {
      if (_edges.at<float>(i, j) <= grad_norm) continue;
      int max_count = 0;
      int orientaion = 0;
      int ori_count[16] = {0};
      for (int dy = -nms_kernel_size / 2; dy < nms_kernel_size / 2 + 1; dy++) {
        for (int dx = -nms_kernel_size / 2; dx < nms_kernel_size / 2 + 1;
             dx++) {
          if (_edges.at<float>(i + dy, j + dx) > grad_norm) {
            int _ori = angle2ori(
                _angles.at<float>(i - rows.start + dy, j - cols.start + dx));
            ori_count[_ori]++;
            if (ori_count[_ori] > max_count) {
              max_count = ori_count[_ori];
              orientaion = _ori;
            }
          }
        }
      }
      if (ori_count[orientaion] > (nms_kernel_size * nms_kernel_size) / 2 + 1) {
        orientation_map.at<uchar>(i - rows.start, j - cols.start) = orientaion;
      }
    }
  }

  // 取领域中的梯度值最大的点为特征点
  Point center(edges.cols / 2, edges.rows / 2);
  for (int i = rows.start; i < rows.end; i++) {
    for (int j = cols.start; j < cols.end; j++) {
      if (edges.at<float>(i, j) <= grad_norm) continue;
      for (int dy = -nms_kernel_size / 2; dy < nms_kernel_size / 2 + 1; dy++) {
        if (edges.at<float>(i, j) <= grad_norm) break;
        for (int dx = -nms_kernel_size / 2; dx < nms_kernel_size / 2 + 1;
             dx++) {
          if (edges.at<float>(i + dy, j + dx) > edges.at<float>(i, j)) {
            edges.at<float>(i, j) = 0.0f;
            break;
          }
        }
      }
      if (edges.at<float>(i, j) > grad_norm) {
        prograds.push_back(
            Features(Point(j - center.x, i - center.y),
                     _angles.at<float>(i - rows.start, j - cols.start),
                     _edges.at<float>(i, j)));
        prograds.back().orientation =
            (int)orientation_map.at<uchar>(i - rows.start, j - cols.start);
      }
    }
  }
}

void Template::scatter(float upper_distance) {
  CV_Assert(upper_distance > 2.0f);
  CV_Assert(!prograds.empty());
  if (prograds.size() < num_features) return;

  float dist = 0.0f;
  float threshold = 0.0f;
  float sum_distance = 0.0f;
  vector<Features> selected_features;

  // 计算每个点的距离分布
  vector<float> distance(prograds.size(), 0.0f);
  for (size_t i = 0; i < prograds.size(); i++) {
    int count_i = 0;
    for (size_t j = 0; j < prograds.size(); j++) {
      if (i == j) continue;
      dist = norm(prograds[i].p_xy - prograds[j].p_xy);
      if (dist < upper_distance) {
        distance[i] += dist;
        count_i++;
      }
    }
    if (!count_i) continue;
    distance[i] /= count_i;
    threshold = fmax(threshold, distance[i]);
    sum_distance += distance[i];
  }

  // 计算每个特征点的选择概率
  std::vector<float> selection_prob(prograds.size());
  for (size_t i = 0; i < prograds.size(); i++) {
    selection_prob[i] = distance[i] / sum_distance;
  }

  // 使用轮盘赌算法选择特征点
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  while (selected_features.size() < num_features) {
    float roulette = dis(gen);
    float cumulative_prob = 0.0f;

    for (size_t i = 0; i < prograds.size(); i++) {
      cumulative_prob += selection_prob[i];
      if (roulette <= cumulative_prob) {
        selected_features.push_back(prograds[i]);
        break;
      }
    }
  }

  prograds = selected_features;
}

vector<Template::Features> Template::relocate_by(
    shapeInfo_producer::Info info) {
  float theta = -info.angle / 180.0f * CV_PI;
  vector<Features> new_prograds;
  for (const auto &pg : prograds) {
    Point newPoint;
    newPoint.x =
        info.scale * (cosf(theta) * pg.p_xy.x - sinf(theta) * pg.p_xy.y);
    newPoint.y =
        info.scale * (sinf(theta) * pg.p_xy.x + cosf(theta) * pg.p_xy.y);
    new_prograds.push_back(
        Features(newPoint, fmod(pg.angle - info.angle, 360.0f), pg.grad_norm));
    new_prograds.back().orientation = angle2ori(new_prograds.back().angle);
  }
  return new_prograds;
}

void Detector::quantize(const cv::Mat &edges, const cv::Mat &angles,
                        cv::Mat &dst, int kernel_size, float gard_norm) {
  Mat ori_bit = Mat::zeros(angles.size(), CV_16U);
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<float>(i, j) <= grad_norm) continue;
      int max_count = 0;
      int orientation = 0;
      int ori_count[16] = {0};
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          if (edges.at<float>(i + dy, j + dx) > grad_norm) {
            ushort _ori = Template::angle2ori(angles.at<float>(i + dy, j + dx));
            ori_count[_ori]++;
            if (ori_count[_ori] > max_count) {
              max_count = ori_count[_ori];
              orientation = _ori;
            }
          }
        }
      }
      if (ori_count[orientation] > (kernel_size * kernel_size) / 2 + 1) {
        ori_bit.at<ushort>(i, j) = 1 << orientation;
      }
    }
  }
}

void Detector::spread(cv::Mat &ori_bit, cv::Mat &spread_ori, int kernel_size) {
  spread_ori = Mat::zeros(ori_bit.size(), CV_16U);
  for (int i = 0; i < ori_bit.rows; i++) {
    for (int j = 0; j < ori_bit.cols; j++) {
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1； dx++) {
          spread_ori.at<ushort>(i + dy, j + dx) |= ori_bit.at<ushort>(i, j);
        }
      }
    }
  }
}

void Detector::computeResponseMaps(cv::Mat &spread_ori,
                                   std::vector<cv::Mat> &response_maps) {
  response_maps.resize(16);
  int maxValue = numeric_limits<ushort>::max();
  for (int orientation = 0; orientation < 16; orientation++) {
    for (int i = 0; i < spread_ori.rows; i++) {
      for (int j = 0; j < spread_ori.cols; j++) {
        response_maps[orientation].at<float>(i, j) =
            cos_table[orientation * maxValue + spread_ori.at<ushort>(i, j)];
      }
    }
  }
}

void Detector::computeSimilarityMap(LinearMemories &memories, Template &temp,
                                    vector<vector<float>> &similarity_map) {
  int n_rows = memories.rows * 4;
  int n_cols = memories.cols * 4;
  // similarity_map[i][j]
  // i -> order in kernel ; j -> index in linear vector
  similarity_map =
      vector<vector<float>>(16, vector<float>(memories.linear_size()));

  for (int i = 0; i < 16; i++) {
    Point start = Point(i%4, i/4);
    for (const auto &pg : temp.pg_ptr()) {
      Point cur = pg.p_xy + start;
      if (cur.x < 0 || cur.x >= n_rows ||
          cur.y < 0 || cur.y >= n_cols)
        continue;
      // 计算坐标在线性存储器中的当前位置
      int position_cur = cur.y / 4 * memories.cols + cur.x / 4;
      // 计算坐标能够移动到的最大位置
      int position_end = (n_rows - 1 - cur.y) / 4 * memories.cols + (n_cols - 1 - cur.x) / 4;
      // 计算坐标在 TxT 分块中的顺序索引
      int order_kernel = (cur.y % 4) * 4 + cur.x % 4;
      // 计算相似度矩阵
      for (int j = position_cur; j <= position_end; j++) {
        similarity_map[i][j] += memories.at(pg.orientation, order_kernel, j);
      }
    }
  }

  // 转化为 100 分制
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < memories.linear_size(); j++) {
      similarity_map[i][j] /= (float)temp.pg_ptr().size();
    }
  }
}

void Detector::localSimilarityMap(LinearMemories &memories, Template &temp,
                                  cv::Mat &similarity_map, cv::Rect roi) {
  int n_rows = memories.rows * 4;
  int n_cols = memories.cols * 4;
  if (!roi.width || !roi.height) 
    roi = Rect(0, 0, n_cols-1, n_rows-1);

  similarity_map = Mat::zeros(roi.height + 1, roi.width + 1, CV_32F);

  for (int i = roi.y; i <= roi.y + roi.height; i++) {
    for (int j = roi.x; j <= roi.x + roi.width; j++) {
      for (const auto &pg : temp.pg_ptr()) {
        Point cur = pg.p_xy + Point(j, i);
        if (cur.y < roi.y || cur.x < roi.x || cur.y > roi.y + roi.height || cur.x > roi.x + roi.width) continue;
        // 计算坐标在线性存储器中的当前位置
        int position_cur = cur.y / 4 * memories.cols + cur.x / 4;
        // 计算坐标在 TxT 分块中的顺序索引
        int order_kernel = (cur.y % 4) * 4 + cur.x % 4;
        // 计算相似度矩阵
        similarity_map.at<float>(i - roi.y, j - roi.x) += memories.at(pg.orientation, order_kernel, position_cur);
      }
    }
  }

  // 转化为 100 分制
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      similarity_map.at<float>(i, j) /= (float)temp.pg_ptr().size();
    }
  }
}

void Detector::linearize(std::vector<cv::Mat> &response_maps,
                         LinearMemories &linearized_memories) {
  int n_rows = response_maps.rows / 4;
  int n_cols = response_maps.cols / 4;
  linearized_memories.resize(16, 16, n_rows * n_cols);
  linearized_memories.rows = n_rows;
  linearized_memories.cols = n_cols;
  for (int orientation = 0; orientation < 16; orientation++) {
    for (int i = 0; i < n_rows; i++) {
      for (int j = 0; j < n_cols; j++) {
        for (int di = 0; di < 3; di++) {
          for (int dj = 0; dj < 3; dj++) {
            int order_in_kernel = di * 4 + dj;
            linearized_memories.at(orientation, order_in_kernel,
                                   i * n_cols + j) =
                response_maps[orientation].at<float>(i + di, j + dj);
          }
        }
      }
    }
  }
}

void Detector::produceRoi(std::vector<MatchPoint> &input_points, cv::Mat &roi, int lower_score, int grid_size) {
  
}

void Detector::setSourceImage(const cv::Mat &src, int pyramid_level,
                              cv::Mat mask) {
  Mat target_src;
  if (!mask.empty())
    src.copyTo(src, mask);
  else
    target_src = src.clone();
  src_pyramid.buildPyramid(target_src, pyramid_level);
}

void Detector::setTempate(const cv::Mat &temp_src, int pyramid_level,
                          Template::TemplateParams params, cv::Mat mask) {
  Mat target_temp;
  if (!mask.empty())
    temp_src.copyTo(target_temp, mask);
  else
    target_temp = temp_src.clone();
  temp_pyramid.buildPyramid(target_temp, pyramid_level);

  Mat downsampled_temp;
  for (int i = 0; i < pyramid_level; i++) {
    Ptr<Template> tp = Template::createPtr_from(temp_pyramid[i], params);
    if (tp->iscreated())
      temps.push_back(tp);
    else
      cerr << "模板创建失败" << endl;
    if (params.nms_kernel_size > 3) params.nms_kernel_size -= 2;
    if (params.num_features > 10)
      params.num_features = params.num_features >> 2;
    if (params.scatter_distance > 2.0f)
      params.scatter_distance = params.scatter_distance / 2.0f;
  }
}

void Detector::match(const cv::Mat &sourceImage, const cv::Mat &templateImage,
                     float lower_score, Template::TemplateParams params,
                     cv::Mat mask_src, cv::Mat mask_temp) {
  int min_CR = min(sourceImage.rows, sourceImage.cols);
  int try_level = 0;
  while (128 * (1 << try_level) < min_CR) try_level++;
  pyramid_level = try_level;
  setSourceImage(sourceImage, pyramid_level, mask_src);
  setTempate(templateImage, pyramid_level, params, mask_temp);

  // 计算梯度响应矩阵
  for (int i = 0; i < pyramid_level; i++) {
    Mat edges, angles;
    Template::getOriMat(src_pyramid[i], edges, angles);

    Mat ori_bit;
    quantize(edges, angles, ori_bit, params.nms_kernel_size, params.grad_norm);

    Mat spread_ori;
    spread(ori_bit, spread_ori, params.nms_kernel_size);

    vector<Mat> response_maps(16, Mat::zeros(src_pyramid[i].size(), CV_32F));
    computeResponseMaps(spread_ori, response_maps);

    linearize(response_maps, memory_pyramid[i]);
  }

  // 从最高层开始相似度匹配
  Mat score_map;
  vector<vector<float>> similarity_map;
  vector<Rect> rois;
  int match_level = pyramid_level - 1;

  computeSimilarityMap(memory_pyramid[match_level], temp_pyramid[match_level], similarity_map);
  produceRoi(similarity_map, rois);

  while (--match_level >= 0) {
    localSimilarityMap(memory_pyramid[match_level], temp_pyramid[match_level], roi);
  }
  
}

void Detector::init_costable() {
  int maxValue = std::numeric_limits<ushort>::max();
  cos_table.resize(16 * maxValue);
  float maxCos;
  for (int i = 0; i < 16; i++) {
    for (ushort bit = 1; bit <= maxValue; bit++) {
      maxCos = 0.0f;
      for (int k = 0; k < 16; k++) {
        if (bit & (1 << k))
          maxCos = maxCos < abs(cos(ori2angle(1 << k) - ori2angle(1 << i)))
                       ? abs(cos(ori2angle(1 << k) - ori2angle(1 << i)))
                       : maxCos;
      }
      cos_table[i * maxValue + bit] = maxCos;
    }
  }
}
