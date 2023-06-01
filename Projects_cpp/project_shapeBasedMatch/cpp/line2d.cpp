#include "line2d.hpp"
using namespace std;
using namespace line2d;

void __onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    cv::Mat *image = static_cast<cv::Mat *>(userdata);
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
  pyramid = vector<cv::Mat>();
}

ImagePyramid::ImagePyramid(const cv::Mat &src, int py_level) {
  buildPyramid(src, py_level);
}

cv::Mat &ImagePyramid::operator[](int index) {
  CV_Assert(index >= 0 && index < pyramid_level);
  return pyramid[index];
}

void ImagePyramid::buildPyramid(const cv::Mat &src, int py_level) {
  pyramid_level = py_level;
  cv::Mat curImg = src.clone();

  // 裁剪图像到合适尺寸
  int suit_size =
      (1 << pyramid_level) * 4;  // 4: 使得最高层图像长宽为 4 的倍数便于线性化
  int n_rows = (curImg.rows / suit_size + 1) * suit_size;
  int n_cols = (curImg.cols / suit_size + 1) * suit_size;
  cv::copyMakeBorder(curImg, curImg, 0, n_rows - curImg.rows, 0,
                     n_cols - curImg.cols, cv::BORDER_REPLICATE);

  // 初始化图像金字塔
  pyramid.push_back(curImg);

  // 构建图像金字塔
  for (int i = 0; i < pyramid_level - 1; i++) {
    cv::Mat downsampledImg;
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

shapeInfo_producer::shapeInfo_producer(const cv::Mat &input_src,
                                       cv::Mat input_mask, bool padding)
    : shapeInfo_producer() {
  CV_Assert(!input_src.empty());
  CV_Assert(input_mask.empty() || (input_src.size() == input_mask.size()));
  CV_Assert(input_mask.empty() || input_mask.type() == CV_8U);
  // 当前图像已经扩充边界
  if (padding) {
    src = input_src;
    if (input_mask.empty())
      mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
    else
      mask = input_mask;
  } else {  // 当前图像未扩充 0 边界
    // 图像在旋转和缩放过程中有效像素到图像中心的最远距离
    int border_max = 1 + (int)sqrt(input_src.rows * input_src.rows +
                                   input_src.cols * input_src.cols);
    // 扩充边界
    cv::copyMakeBorder(input_src, src, border_max - input_src.rows / 2,
                       border_max - input_src.rows / 2,
                       border_max - input_src.cols / 2,
                       border_max - input_src.cols / 2, cv::BORDER_REPLICATE);

    if (input_mask.empty())
      mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
    else  // 扩充掩图边界
      cv::copyMakeBorder(input_mask, mask, border_max - input_mask.rows / 2,
                         border_max - input_mask.rows / 2,
                         border_max - input_mask.cols / 2,
                         border_max - input_mask.cols / 2, cv::BORDER_CONSTANT);
  }
}

void shapeInfo_producer::save_config(const shapeInfo_producer &sip,
                                     string path) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);

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

cv::Ptr<shapeInfo_producer> shapeInfo_producer::load_config(
    const cv::Mat &input_src, cv::Mat input_mask, bool padding, string path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);

  // 用 Ptr 管理并初始化 shapeInfo_producer 对象
  cv::Ptr<shapeInfo_producer> sip =
      cv::makePtr<shapeInfo_producer>(input_src, input_mask, padding);

  // 读取公有成员参数
  cv::FileNode paramsNode = fs["params"];
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

cv::Mat shapeInfo_producer::affineTrans(const cv::Mat &src, float angle,
                                        float scale) {
  cv::Mat dst;
  cv::Point2f center(cvFloor(src.cols / 2.0f), cvFloor(src.rows / 2.0f));
  cv::Mat rotate_mat = cv::getRotationMatrix2D(center, angle, scale);
  cv::warpAffine(src, dst, rotate_mat, src.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  return dst;
}

cv::Mat shapeInfo_producer::src_of(const Info &info) {
  return affineTrans(src, info.angle, info.scale);
}

cv::Mat shapeInfo_producer::mask_of(const Info &info) {
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

Template::Template() {
  nms_kernel_size = defaultParams.nms_kernel_size;
  scatter_distance = defaultParams.scatter_distance;
  grad_norm = defaultParams.grad_norm;
  num_features = defaultParams.num_features;
  template_created = defaultParams.template_created;
}

Template::Template(TemplateParams params, bool isDefault) {
  nms_kernel_size = params.nms_kernel_size;
  scatter_distance = params.scatter_distance;
  grad_norm = params.grad_norm;
  num_features = params.num_features;
  template_created = params.template_created;
  if (isDefault) defaultParams = params;
}

void Template::getOriMat(const cv::Mat &src, cv::Mat &edges, cv::Mat &angles) {
  CV_Assert(!src.empty());
  CV_Assert(src.channels() == 1 || src.channels() == 3);

  /// @variables:
  vector<cv::Mat> imgs(3);
  split(src, imgs);  // 拆分成三个通道
  cv::Mat gx = cv::Mat::zeros(src.size(), CV_32F);
  cv::Mat gy = cv::Mat::zeros(src.size(), CV_32F);
  edges = cv::Mat::zeros(src.size(), CV_32F);
  angles = cv::Mat::zeros(src.size(), CV_32F);
  float max_grad_norm = 0.0f;

  // 计算梯度模长和方向角
  if (src.channels() == 1) {  // src 为单通道图像
    // cv::GaussianBlur(imgs[0], imgs[0], Size(5, 5), 1.5);
    cv::Scharr(imgs[0], gx, CV_32F, 1, 0, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Scharr(imgs[0], gy, CV_32F, 0, 1, 1.0, 0.0, cv::BORDER_REPLICATE);
    for (int i = 0; i < edges.rows; i++) {
      for (int j = 0; j < edges.cols; j++) {
        edges.at<float>(i, j) = sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                                     gy.at<float>(i, j) * gy.at<float>(i, j));
        angles.at<float>(i, j) =
            cv::fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));
        if (max_grad_norm < edges.at<float>(i, j))
          max_grad_norm = edges.at<float>(i, j);
      }
    }
  } else {  // src 为 3 通道图像
    vector<cv::Mat> cedges(3, cv::Mat::zeros(src.size(), CV_32F));
    vector<cv::Mat> cangles(3, cv::Mat::zeros(src.size(), CV_32F));
    for (int k = 0; k < 3; k++) {
      // GaussianBlur(imgs[k], imgs[k], Size(5, 5), 1.5);
      cv::Scharr(imgs[k], gx, CV_32F, 1, 0, 1.0, 0.0, cv::BORDER_REPLICATE);
      cv::Scharr(imgs[k], gy, CV_32F, 0, 1, 1.0, 0.0, cv::BORDER_REPLICATE);
      for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
          cedges[k].at<float>(i, j) =
              sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                   gy.at<float>(i, j) * gy.at<float>(i, j));
          cangles[k].at<float>(i, j) =
              cv::fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));
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
  cv::Mat edges, angles;

  getOriMat(src, edges, angles);
  // cv::Mat edgesImage;
  // normalize(edges, edgesImage, 0, 255, NORM_MINMAX, CV_8U);
  // imshow("edges", edgesImage);
  // waitKey();

  tp.selectFeatures_from(edges, angles, nms_kernel_size);

  tp.scatter(scatter_distance);
}

cv::Ptr<Template> Template::createPtr_from(const cv::Mat &src,
                                           TemplateParams params) {
  cv::Ptr<Template> tp = cv::makePtr<Template>(params);

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

void line2d::Template::create_from(const cv::Mat &src) {
  for (int i = 0; i < 100; i++) {
    createTemplate(src, *this, nms_kernel_size, scatter_distance);
    if (prograds.size() >= num_features)
      break;
    else {
      if (nms_kernel_size > 3)
        nms_kernel_size -= 2;
      else {
        scatter_distance +=
            (float)nms_kernel_size * num_features / prograds.size();
      }
    }
  }
  if (prograds.size() >= num_features)
    template_created = true;
  else {
    cerr << "无法找到的足够的特征点！" << endl;
    prograds.clear();
  }
}

void Template::selectFeatures_from(const cv::Mat &_edges,
                                   const cv::Mat &_angles,
                                   int nms_kernel_size) {
  cv::Mat edges = _edges.clone();
  cv::Mat angles = cv::Mat::zeros(_angles.size(), CV_8U);

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
  // cv::Range rows(nms_kernel_size / 2, nms_kernel_size / 2 + edges.rows);
  // cv::Range cols(nms_kernel_size / 2, nms_kernel_size / 2 + edges.cols);
  // cv::copyMakeBorder(edges, edges, nms_kernel_size / 2, nms_kernel_size / 2,
  //                    nms_kernel_size / 2, nms_kernel_size / 2,
  //                    cv::BORDER_CONSTANT, cv::Scalar(0));
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      float prevPixel, nextPixel;
      switch ((int)angles.at<uchar>(i, j)) {
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
  // cv::Mat edgesImage;
  // cv::normalize(edges, edgesImage, 0, 255, cv::NORM_MINMAX, CV_8U);
  // cv::namedWindow("edges after nms", cv::WINDOW_NORMAL);
  // cv::imshow("edges after nms", edges);
  // cv::waitKey();
  // cv::namedWindow("angles after quantized", cv::WINDOW_NORMAL);
  // cv::imshow("angles after quantized", angles);
  // cv::waitKey();

  // 计算领域中出现次数最多的量化特征方向
  cv::Mat orientation_map = cv::Mat::zeros(_angles.size(), CV_8U);
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (_edges.at<float>(i, j) <= grad_norm) continue;
      int max_count = 0;
      int orientaion = 0;
      int ori_count[16] = {0};
      for (int dy = -nms_kernel_size / 2; dy < nms_kernel_size / 2 + 1; dy++) {
        for (int dx = -nms_kernel_size / 2; dx < nms_kernel_size / 2 + 1;
             dx++) {
          if (_edges.at<float>(i + dy, j + dx) > grad_norm) {
            int _ori = angle2ori(_angles.at<float>(i + dy, j + dx));
            ori_count[_ori]++;
            if (ori_count[_ori] > max_count) {
              max_count = ori_count[_ori];
              orientaion = _ori;
            }
          }
        }
      }
      orientation_map.at<uchar>(i, j) = orientaion;
    }
  }
  // cv::namedWindow("orientation_map", cv::WINDOW_NORMAL);
  // cv::imshow("orientation_map", orientation_map * 16);
  // cv::waitKey();

  // 取领域中的梯度值最大的点为特征点
  cv::Point center(edges.cols / 2, edges.rows / 2);
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
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
        prograds.push_back(Features(cv::Point(j - center.x, i - center.y),
                                    _angles.at<float>(i, j),
                                    _edges.at<float>(i, j)));
        prograds.back().orientation = (int)orientation_map.at<uchar>(i, j);
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
      dist = cv::norm(prograds[i].p_xy - prograds[j].p_xy);
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
  vector<float> selection_prob(prograds.size());
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
    cv::Point newPoint;
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

void Template::show_in(const cv::Mat &background, cv::Point center) {
  cv::Mat templateImage;
  if (background.type() == CV_8U)
    cv::cvtColor(background, templateImage, cv::COLOR_GRAY2BGR);
  else
    templateImage = background.clone();

  for (const auto &pg : prograds) {
    cv::circle(templateImage, center + pg.p_xy, 1, cv::Scalar(0, 0, 255), -1);
  }
  cv::circle(templateImage, center, 1, cv::Scalar(0, 255, 0), -1);

  cv::namedWindow("templateImage", cv::WINDOW_NORMAL);
  cv::imshow("templateImage", templateImage);
  cv::waitKey();
}

Detector::Detector() {
  pyramid_level = 2;
  init_costable();
}

void Detector::quantize(const cv::Mat &edges, const cv::Mat &angles,
                        cv::Mat &ori_bit, int kernel_size, float grad_norm) {
  ori_bit = cv::Mat::zeros(angles.size(), CV_16U);
  // cv::Mat ori_mat = cv::Mat::ones(angles.size(), CV_8U);
  // ori_mat *= 125;
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<float>(i, j) <= grad_norm) continue;
      int max_count = 0;
      int orientation = 0;
      int ori_count[16] = {0};
      for (int dy = -kernel_size / 2; dy < kernel_size / 2 + 1; dy++) {
        for (int dx = -kernel_size / 2; dx < kernel_size / 2 + 1; dx++) {
          if (i + dy < 0 || i + dy >= edges.rows || j + dx < 0 ||
              j + dx >= edges.cols)
            continue;
          if (edges.at<float>(i + dy, j + dx) > grad_norm) {
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
  // cv::namedWindow("ori_mat", cv::WINDOW_NORMAL);
  // cv::imshow("ori_mat", ori_mat);
  // cv::waitKey();
}

void Detector::spread(cv::Mat &ori_bit, cv::Mat &spread_ori, int kernel_size) {
  spread_ori = cv::Mat::zeros(ori_bit.size(), CV_16U);
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

void Detector::computeResponseMaps(cv::Mat &spread_ori,
                                   vector<cv::Mat> &response_maps) {
  response_maps.resize(16);
  for (int i = 0; i < 16; i++)
    response_maps[i] = cv::Mat::zeros(spread_ori.size(), CV_32F);

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

void Detector::computeSimilarityMap(vector<LinearMemories> &memories,
                                    Template &temp,
                                    LinearMemories &similarity) {
  // for (int j = 0; j < 16; j++) {
  //   cv::Mat resImage, res_map;
  //   unlinearize(memories[j], res_map);
  //   resImage = (res_map + 1) * (255.0f / 2);
  //   resImage.convertTo(resImage, CV_8U);
  //   temp.show_in(resImage, cv::Point(resImage.cols/2, resImage.rows/2));
  // }
  // similarity[i][j]
  // i -> order in kernel ; j -> index in linear vector
  similarity.create(16, memories[0].linear_size(), 0.0f);
  similarity.rows = memories[0].rows;
  similarity.cols = memories[0].cols;
  // cout << similarity.rows << "x" << similarity.cols << ":"
  //      << similarity.linear_size() << endl;
  // cin.get();

  for (int i = 0; i < 16; i++) {
    cv::Point start = cv::Point(i % 4, i / 4);
    for (const auto &pg : temp.pg_ptr()) {
      cv::Point cur = pg.p_xy + start;
      cv::Range range_x, range_y;
      if (cur.x < 0)
        range_x = cv::Range( (-cur.x) / 4, similarity.cols);
      else
        range_x = cv::Range(0, (similarity.cols * 4 - cur.x - 1) / 4);

      if (cur.y < 0)
        range_y = cv::Range( (-cur.y) / 4, similarity.rows);
      else
        range_y = cv::Range(0, (similarity.rows * 4 - cur.y - 1) / 4);

      int mod_y = cur.y % 4 < 0 ? (cur.y % 4) + 4 : cur.y % 4;
      int mod_x = cur.x % 4 < 0 ? (cur.x % 4) + 4 : cur.x % 4;

      // cout << "(debug)" << endl;
      // cout << "坐标:" << cur.y << "," << cur.x << endl;
      // cout << "order in kernel:" << order_kernel << endl;
      // cout << "cur position in linear vector:" << position_cur << endl;
      // cout << "end position in linear vector:" << position_end << endl;
      // cin.get();

      // 计算相似度矩阵
      for (int y = range_y.start; y < range_y.end; y++) {
        for (int x = range_x.start; x < range_x.end; x++) {
          similarity.at(i, y * similarity.cols + x) +=
              memories[pg.orientation].at(mod_y * 4 + mod_x,
                                          y * similarity.cols + x);
        }
      }
    }
  }

  // 转化为 100 分制
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < memories[0].linear_size(); j++) {
      similarity.at(i, j) =
          similarity.at(i, j) / (float)temp.pg_ptr().size() * 100.0f;
    }
  }
}

void Detector::localSimilarityMap(vector<LinearMemories> &memories,
                                  Template &temp, cv::Mat &similarity_map,
                                  vector<cv::Rect> &rois) {
  // CV_Assert(!rois.empty());
  int n_rows = memories[0].rows * 4;
  int n_cols = memories[0].cols * 4;
  if (rois.empty()) {
    rois.push_back(cv::Rect(0, 0, n_cols-1, n_rows-1));
  }

  similarity_map = cv::Mat::zeros(n_rows, n_cols, CV_32F);

  for (size_t k = 0; k < rois.size(); k++) {
    for (int i = rois[k].y; i <= rois[k].y + rois[k].height; i++) {
      for (int j = rois[k].x; j <= rois[k].x + rois[k].width; j++) {
        for (const auto &pg : temp.pg_ptr()) {
          cv::Point cur = pg.p_xy + cv::Point(j, i);

          if (cur.y < rois[k].y || cur.x < rois[k].x ||
              cur.y > rois[k].y + rois[k].height ||
              cur.x > rois[k].x + rois[k].width)
            continue;

          // 计算坐标在线性存储器中的当前位置
          int position_cur = cur.y / 4 * memories[0].cols + cur.x / 4;
          // 计算坐标在 TxT 分块中的顺序索引
          int order_kernel = (cur.y % 4) * 4 + cur.x % 4;
          // 计算相似度矩阵
          similarity_map.at<float>(i, j) +=
              memories[pg.orientation].at(order_kernel, position_cur);
        }
      }
    }
  }

  // 转化为 100 分制
  similarity_map = similarity_map / (float)temp.pg_ptr().size() * 100.0f;
}

void Detector::linearize(std::vector<cv::Mat> &response_maps,
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
                           cv::Mat &similarity_map) {
  similarity_map =
      cv::Mat::zeros(similarity.rows * 4, similarity.cols * 4, CV_32F);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < similarity.linear_size(); j++) {
      int u = (j / similarity.cols) * 4 + i / 4;
      int v = (j % similarity.cols) * 4 + i % 4;
      similarity_map.at<float>(u, v) = similarity.at(i, j);
    }
  }
}

void Detector::produceRoi(cv::Mat &similarity_map,
                          std::vector<cv::Rect> &roi_list, int lower_score) {
  cv::Mat binary_mat;
  cv::threshold(similarity_map, binary_mat, lower_score, 100.0,
                cv::THRESH_BINARY);
  cv::namedWindow("binaryImage", cv::WINDOW_NORMAL);
  cv::imshow("binaryImage", binary_mat);
  cv::waitKey();

  cv::Mat labels, stats, centroids;
  int numLabels =
      cv::connectedComponentsWithStats(binary_mat, labels, stats, centroids);

  roi_list.clear();
  for (int i = 1; i < numLabels; i++) {
    cv::Rect roi(stats.at<int>(i, cv::CC_STAT_LEFT),
                 stats.at<int>(i, cv::CC_STAT_TOP),
                 stats.at<int>(i, cv::CC_STAT_WIDTH),
                 stats.at<int>(i, cv::CC_STAT_HEIGHT));
    roi_list.push_back(roi);
  }
}

void Detector::setSourceImage(const cv::Mat &src, int pyramid_level,
                              cv::Mat mask) {
  cv::Mat target_src;
  if (!mask.empty())
    src.copyTo(src, mask);
  else
    target_src = src.clone();
  src_pyramid.buildPyramid(target_src, pyramid_level);
}

void Detector::setTempate(const cv::Mat &temp_src, int pyramid_level,
                          Template::TemplateParams params, cv::Mat mask) {
  cv::Mat target_temp;
  if (!mask.empty())
    temp_src.copyTo(target_temp, mask);
  else
    target_temp = temp_src.clone();
  temp_pyramid.buildPyramid(target_temp, pyramid_level);

  for (int i = 0; i < pyramid_level; i++) {
    cv::Ptr<Template> tp = Template::createPtr_from(temp_pyramid[i], params);
    tp->show_in(temp_pyramid[i],
                cv::Point(temp_pyramid[i].cols / 2, temp_pyramid[i].rows / 2));
    if (tp->iscreated())
      temps.push_back(tp);
    else
      cerr << "模板创建失败" << endl;

    // 更新下一层构造模板所需的参数
    if (params.nms_kernel_size > 3) params.nms_kernel_size -= 2;
    if (params.num_features > 10)
      params.num_features = params.num_features >> 2;
    if (params.scatter_distance > 2.0f)
      params.scatter_distance = params.scatter_distance / 2.0f;
  }
}

void Detector::selectMatchPoints(cv::Mat &similarity_map,
                                 vector<cv::Rect> &roi_list, int lower_score) {
  CV_Assert(!roi_list.empty());
  match_points.clear();

  float tol_score = 0.0f;
  cv::Point2f best_match;

  for (const cv::Rect &roi : roi_list) {
    best_match = cv::Point2f(0.0f, 0.0f);
    tol_score = 0.0f;
    for (int x = roi.x; x <= roi.x + roi.width; x++) {
      for (int y = roi.y; y <= roi.y + roi.height; y++) {
        float &score = similarity_map.at<float>(y, x);
        if (score > lower_score) {
          tol_score += score;
          best_match += cv::Point2f(score * x, score * y);
        }
      }
    }
    if (tol_score > 0.0f) {
      best_match.x /= tol_score;
      best_match.y /= tol_score;
      float &score =
          similarity_map.at<float>((int)best_match.y, (int)best_match.x);
      match_points.push_back(MatchPoint(cv::Point(best_match), score));
    }
  }
}

void Detector::match(const cv::Mat &sourceImage, const cv::Mat &templateImage,
                     int lower_score, Template::TemplateParams params,
                     cv::Mat mask_src, cv::Mat mask_temp) {
  int min_CR = min(sourceImage.rows, sourceImage.cols);
  int try_level = 0;
  while (128 * (1 << try_level) < min_CR) try_level++;
  pyramid_level = try_level;
  setSourceImage(sourceImage, pyramid_level, mask_src);
  setTempate(templateImage, pyramid_level, params, mask_temp);

  // 计算梯度响应矩阵
  memory_pyramid.resize(pyramid_level);
  for (int i = 0; i < pyramid_level; i++) {
    cv::Mat edges, angles;
    Template::getOriMat(src_pyramid[i], edges, angles);
    // cv::Mat edgesImage;
    // cv::normalize(edges, edgesImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("edges", edgesImage);
    // cv::waitKey();

    cv::Mat ori_bit;
    quantize(edges, angles, ori_bit, params.nms_kernel_size, params.grad_norm);

    cv::Mat spread_ori;
    spread(ori_bit, spread_ori, params.nms_kernel_size);
    // cv::Mat spreadImage;
    // cv::namedWindow("spreadImage", cv::WINDOW_NORMAL);
    // cv::normalize(spread_ori, spreadImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("spreadImage", spreadImage);
    // cv::waitKey();

    vector<cv::Mat> response_maps;
    computeResponseMaps(spread_ori, response_maps);
    // for (int i = 0; i < 16; i++) {
    //   cv::Mat responseImage;
    //   cv::namedWindow("responseImage " + to_string(i + 1),
    //   cv::WINDOW_NORMAL); responseImage = (response_maps[i] + 1) * (255.0f /
    //   2); responseImage.convertTo(responseImage, CV_8U);
    //   cv::imshow("responseImage " + to_string(i + 1), responseImage);
    //   cv::waitKey();
    // }

    linearize(response_maps, memory_pyramid[i]);

    // for (int j = 0; j < 16; j++) {
    //   cv::Mat resImage, res_map;
    //   unlinearize(memory_pyramid[i][j], res_map);
    //   cv::namedWindow("resImage " + to_string(j + 1), cv::WINDOW_NORMAL);
    //   resImage = (res_map + 1) * (255.0f / 2);
    //   resImage.convertTo(resImage, CV_8U);
    //   cv::imshow("resImage " + to_string(j + 1), resImage);
    //   cv::waitKey();
    // }
  }

  // 从最高层开始相似度匹配
  vector<cv::Rect> rois;
  cv::Mat similarity_map;
  LinearMemories similarity;
  int match_level = pyramid_level - 1;

  computeSimilarityMap(memory_pyramid[match_level], *temps[match_level],
                       similarity);

  unlinearize(similarity, similarity_map);
  cv::Mat similarityImage;
  cv::namedWindow("similarityImage", cv::WINDOW_NORMAL);
  cv::normalize(similarity_map, similarityImage, 0, 255, cv::NORM_MINMAX,
                CV_8U);
  cv::imshow("similarityImage", similarityImage);
  cv::setMouseCallback("similarityImage", __onMouse, &similarity_map);
  cv::waitKey();

  localSimilarityMap(memory_pyramid[match_level], *temps[match_level],
                       similarity_map, rois);
  cv::namedWindow("localsimilarityImage", cv::WINDOW_NORMAL);
  cv::normalize(similarity_map, similarityImage, 0, 255, cv::NORM_MINMAX,
                CV_8U);
  cv::imshow("localsimilarityImage", similarityImage);
  cv::setMouseCallback("localsimilarityImage", __onMouse, &similarity_map);
  cv::waitKey();
      

  produceRoi(similarity_map, rois, lower_score);

  while (--match_level >= 0) {
    vector<cv::Rect> next_rois;

    for (const cv::Rect &roi : rois) {
      cv::Rect next_roi(roi.x * 2, roi.y * 2, roi.width * 2, roi.height * 2);
      next_rois.push_back(next_roi);
    }

    rois = next_rois;

    localSimilarityMap(memory_pyramid[match_level], *temps[match_level],
                       similarity_map, rois);

    produceRoi(similarity_map, rois, lower_score);
  }

  selectMatchPoints(similarity_map, rois, lower_score);
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
          // cout << "(debug)" << endl;
          // cout << "i:" << bitset<16>(1<<i) << endl;
          // cout << "bit:" << bitset<16>(bit) << endl;
          // cout << "k:" << bitset<16>(1<<k) << endl;
          // cout << "ori(i)" << ori2angle(i) << endl;
          // cout << "ori(k)" << ori2angle(k) << endl;
          // cout << "cos(ori(i) - oir(k)):" << cos(abs(i - k) *
          // _degree_(11.25f)) << endl; cin.get();
        }
      }
      cos_table[i * maxValue + bit] = maxCos;
    }
    // time.out(to_string(i+1)+"轮循环 65535 * 16 计算结束");
    // cin.get();
  }
}
