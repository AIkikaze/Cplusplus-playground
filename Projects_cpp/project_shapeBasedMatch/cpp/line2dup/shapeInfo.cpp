#include "shapeInfo.hpp"
using namespace std;

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

cv::Mat shapeInfo_producer::src_of(Info info) {
  return affineTrans(src, info.angle, info.scale);
}

cv::Mat shapeInfo_producer::mask_of(Info info) {
  return affineTrans(mask, info.angle, info.scale);
}

const vector<shapeInfo_producer::Info> &shapeInfo_producer::Infos_constptr()
    const {
  return Infos;
}