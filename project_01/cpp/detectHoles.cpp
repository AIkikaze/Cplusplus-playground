#include "detectHoles.hpp"
using namespace cv;
using namespace std;

HoleDetector::HoleDetector() {
  params.size_blur = 13;
  params.size_adaptive_binary_block = 53;
  params.c_adaptive_const = 2.0;
  params.size_structeElement = 7;
  params.times_morphologyEx = 7;
  params.type_morphologyEx = MORPH_CLOSE;
  params.size_holeImage = 320;
  selecting = false;
  afterProcess = false;
  holeFound = false;
}

void HoleDetector::setImage(const Mat &image) {
  pImage = image.clone();
}

void HoleDetector::processImage() {
  // 判断已经设置好灰度图片
  CV_Assert(!pImage.empty());
  CV_Assert(pImage.type() == CV_8U);
  // 盒式滤波
  blur(pImage, pImage, Size(params.size_blur, params.size_blur));
  // 自适应二值化
  adaptiveThreshold(pImage, pImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, params.size_adaptive_binary_block, params.c_adaptive_const);
  // 定义结构元素
  Mat kernel = getStructuringElement(MORPH_RECT, Size(params.size_structeElement, params.size_structeElement));
  // 执行闭运算
  for(int i = 0; i < params.times_morphologyEx; i++) {
    morphologyEx(pImage, pImage, params.type_morphologyEx, kernel);
  }
  // 存储彩色图像
  cvtColor(pImage, holeImage, COLOR_GRAY2BGR);
  // 标记图像处理已完成
  afterProcess = true;
}

void HoleDetector::drawResult() {
  // 判断已经完成图像处理
  CV_Assert(afterProcess);
  namedWindow("resultImage", WINDOW_NORMAL);
  imshow("resultImage", pImage);
  cout << "等待任意键...关闭图像窗口" << endl;
  waitKey();
  destroyWindow("resultImage");
}

void HoleDetector::drawHoleList(const cv::String &winname, int flag = 0) {
  // 显示当前孔洞序列
  CV_Assert(holelist.size());
  // 创建画布
  Mat canvas(params.size_holeImage, params.size_holeImage * holelist.size(), CV_8UC3, Scalar(0, 0, 0));
  // 孔洞图片复制到画布上
  int addcols = 0;
  for(const auto &hole : holelist_image) {
    hole.copyTo(canvas(Rect(addcols, 0, params.size_holeImage, params.size_holeImage)));
    addcols += params.size_holeImage;
  }
  // 显示图片
  namedWindow(winname, WINDOW_AUTOSIZE);
  imshow(winname, canvas);
  // 输出定位孔序列
  cout << "定位孔位置：" << endl;
  for(auto &i : holelist) 
    cout << i << endl;
  if(!flag) {
    waitKey();
    destroyWindow(winname);
  }
}

void HoleDetector::addNewHole() {
  // 判断已经完成图像处理
  CV_Assert(afterProcess);
  CV_Assert(pImage.type() == CV_8U);
  // 初始化 holefound
  holeFound = false;
  // 创建检测孔洞窗口
  namedWindow("holedetect", WINDOW_NORMAL);
  // 显示图片
  imshow("holedetect", pImage);
  // 注册鼠标相应函数
  __setMouseCallback("holedetect");
  // 等待键盘
  waitKey();
}

void HoleDetector::copyHolelistTo(vector<Point> &dst) {
  dst = holelist;
}

void HoleDetector::copyHolelistImageTo(std::vector<cv::Mat> &dst) {
  dst = holelist_image;
}

int HoleDetector::getSize() {
  return holelist.size();
}

void HoleDetector::__setMouseCallback(const String &winname) {
  setMouseCallback(winname, &HoleDetector::onMouse, this);
}

void HoleDetector::onMouse(int event, int x, int y, int flags, void *userdata) {
  HoleDetector* self = static_cast<HoleDetector*>(userdata);
  // 判断是否已经找到定位孔
  if(self->holeFound) {
    // 结束
    cout << "已成功找到定位圆孔!" << endl;
    cout << "等待任意键...关闭图像窗口" << endl;
    setMouseCallback("holedetect", nullptr, nullptr);
    return;
  }
  // 记录当前点的位置
  if(self->selecting) {
    self->end_point = Point(x, y);
    self->box_selecting = Rect(self->start_point, self->end_point);
  }
  else 
    self->start_point = Point(x, y);
  // 判断框选条件
  // 选框未启用 + 鼠标左键按下
  if(!self->selecting && event == EVENT_LBUTTONDOWN) {
    self->selecting = true;
    self->start_point = Point(x, y);
    self->box_selecting = Rect(x, y, 0, 0);
    cout << "------------- 开始框选 roi 区域 ----------------" << endl;
    cout << "roi 区域左上角顶点已选定！其坐标为：" << x << "," << y << endl;
  }
  // 选框已启用 + 鼠标左键释放
  else if(self->selecting && event == EVENT_LBUTTONUP) {
    self->end_point = Point(x, y);
    self->selecting = false;
    self->roiImage  = self->pImage(self->box_selecting);
    namedWindow("roi");
    imshow("roi", self->roiImage);
    self->findHoles();
    if(self->holeFound) self->drawHoleList("Holes Found!");
    waitKey();
    destroyWindow("roi");
  }
  // 选框已启用 + 鼠标移动
  else if(self->selecting && event == EVENT_MOUSEMOVE) {
    Mat tempImage;
    cvtColor(self->pImage, tempImage, COLOR_GRAY2BGR);
    rectangle(tempImage, self->box_selecting, Scalar(0, 0, 255), 13);
    imshow("holedetect", tempImage);
  }
}

vector<Vec6d> HoleDetector::detectEllipse(const cv::Mat &grey, const double &scale) {
  Mat temp;
  // 图像缩放
  if (scale < 1.0)
    resize(grey, temp, cv::Size(grey.cols * scale, grey.rows * scale));
  else
    temp = grey;

  // 创建 edge drawing 对象 ed 用于检测椭圆
  cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
  ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::PREWITT;
  ed->params.GradientThresholdValue = 20;
  ed->params.AnchorThresholdValue = 2;

  // 检测边缘
  ed->detectEdges(temp);
  vector<cv::Vec4d> lines;
  // 检测直线，检测直线前需边缘检测
  ed->detectLines(lines);
  // 检测圆与椭圆，检测圆前需要检测直线
  vector<Vec6d> ellipses;
  ed->detectEllipses(ellipses);

  if (scale < 1.0) {
    // 还原椭圆位置
    double s = 1.0 / scale;
    for (auto &vec : ellipses) {
      // 还原椭圆尺寸
      for (int i = 0; i < 5; i++)
        vec[i] *= s;
    }
  }
  return ellipses;
}

void HoleDetector::findHoles() {
  // 创建孔洞图像
  CV_Assert(!holeImage.empty());
  CV_Assert(!roiImage.empty());
  CV_Assert(holeImage.type() == CV_8UC3);
  CV_Assert(roiImage.type() == CV_8U);
  // 创建圆心序列
  vector<Point2d> center_list;
  // 圆形检测
  vector<Vec6d> ellipses = detectEllipse(roiImage, 1.0);
  // 绘制圆形 
  for(size_t i = 0; i < ellipses.size(); i++) {
    Point center((int)ellipses[i][0], (int)ellipses[i][1]);
    Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
    double angle(ellipses[i][5]);
    Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);
    // 过滤半径小于 30 的圆
    if (ellipses[i][2] && ellipses[i][2] < 30) 
      continue;
    // 过滤某一轴长度小于 30 的椭圆
    else if (!ellipses[i][2] && (ellipses[i][3] < 30 || ellipses[i][4] < 30)) 
      continue;
    center_list.push_back(Point2d(ellipses[i][0], ellipses[i][1]));
    center += start_point;
    // 在图像上绘制圆心
    circle(holeImage, center, 5, Scalar(0, 0, 255), -1);
    // 在图像上绘制圆，或椭圆形
    ellipse(holeImage, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
  }
  // 判断同心圆
  bool flag = areCirclesApproximatelyConcentric(center_list, 2.0);
  if(!flag) {
    cout << "请重新绘制 roi 区域..." << endl;
    return;
  }
  // 储存孔洞图像
  Point center_star = holelist.back();
  int min_dist1 = min(center_star.x, center_star.y);
  int min_dist2 = min(pImage.cols-1-center_star.x, pImage.rows-1-center_star.y);
  int min_dist = min(min(min_dist1, min_dist2), params.size_holeImage*4);
  Point p_leftup = center_star - Point(min_dist, min_dist);
  Point p_rightdown = center_star + Point(min_dist, min_dist);
  Mat resizedImage = holeImage(Rect(p_leftup, p_rightdown));
  // 缩放图片
  resize(resizedImage, resizedImage, Size(params.size_holeImage, params.size_holeImage));
  holelist_image.push_back(resizedImage);
  // 成功找到定位圆孔
  holeFound = true;
}

bool HoleDetector::areCirclesApproximatelyConcentric(const vector<Point2d> &circles, double deviationThreshold) {
  if (circles.empty()) {
    cout << "错误：未找到圆！" << endl;
    return false;
  }

  // 计算圆心的平均值
  Point2d averageCenter(0, 0);
  for (const auto &circle : circles) {
    averageCenter += circle;
  }
  averageCenter /= (double)(circles.size());

  // 计算圆心的偏差
  double deviation = 0.0;
  for (const auto &circle : circles) {
    deviation += cv::norm(circle - averageCenter);
  }
  deviation /= (double)(circles.size());

  Point resulCenter = start_point + (Point)averageCenter;

  cout << "当前平均圆心：" << resulCenter << endl;
  cout << "当前标准差：" << deviation << endl;
  if (deviation > deviationThreshold) {
    cout << "错误：找到的圆没有近似同心！" << endl;
    return false;
  }
  else {
    cout << "成功：所有圆心偏差小于阈值，近似同心" << endl;
   
    holelist.push_back(resulCenter);
  }
  return true;
}