#include "detectHoles.hpp"
using namespace cv;
using namespace std;

HoleDetector::HoleDetector() {
  params.size_blur = 13;
  params.size_adaptive_binary_block = 53;
  params.c_adaptive_const = 2.0;
  params.size_structeElement = 5;
  params.times_morphologyEx = 9;
  params.type_morphologyEx = MORPH_CLOSE;
  params.select_roi = false;
  params.afterProcess = false;
  params.size_holeImage = 128;
}

HoleDetector::setImage(const Mat &image) {
  pImage = image.clone();
}

HoleDetector::processImage() {
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
  // 标记图像处理已完成
  afterProcess = true;
}

HoleDetector::drawResult() {
  // 判断已经完成图像处理
  CV_Assert(afterProcess);
  namedWindow("resultImage", WINDOW_NORMAL);
  imshow("resultImage", pImage);
  cout << "等待任意键...关闭图像窗口" << endl;
  waitKey();
}

HoleDetector::drwaHoleList() {
  // 显示当前孔洞序列
  CV_Assert(holelist.size());
  // 创建画布
  Mat canvas(params.size_holeImage, params.size_holeImage * holelist.size(), CV_8UC3, Scalar(0, 0, 0));
  // 创建窗口对象
  namedWindow("holelist", WINDOW_NORMAL);
  // 孔洞图片复制到画布上
  int addcols = 0;
  for(const auto &hole : holelist_image) {
    hole.copyTo(canvas(Rect(addcols, 0, params.size_holeImage, params.size_holeImage)));
    addcols += params.size_holeImage;
  }
  // 显示图片
  imshow("holelist", canvas);
  cout << "等待任意键...关闭图像窗口" << endl;
  waitKey();
}

HoleDetector::addNewHole() {
  // 判断已经完成图像处理
  CV_Assert(afterProcess);
  CV_Assert(pImage.type() == CV_8U);
  // 创建检测孔洞窗口
  namedWindow("holedetect", WINDOW_NORMAL);
  // 将灰度图转化为彩色图，以便于图像显示
  Mat colorImage;
  cvtColor(pImage, colorImage, COLOR_GRAY2BGR);
  // 显示图片
  imshow("holedetect", pImage)
  // 注册鼠标相应函数
  setMouseCallback("holedetect", HoleDetector::onMouse, &)
  // 结束
  cout << "等待任意键...关闭图像窗口" << endl;
  waitKey();
}

HoleDetector::onMouse(int event, int x, int y, int flags, void *userdata) {
  // 记录当前点的位置
  if(selecting) {
    end_point = Point(x, y);
    box_selecting = Rect(start_point, end_point);
  }
  else 
    start_point = Point(x, y);
  // 判断框选条件
  // 选框未启用 + 鼠标左键按下
  if(!selecting && event == EVENT_LBUTTONDOWN) {
    selecting = true;
    start_point = Point(x, y);
    box_selecting = Rect(x, y, 0, 0);
    cout << "roi 区域左上角顶点已选定！其坐标为：" << x << "," << y << endl;
  }
  // 选框已启用 + 鼠标左键释放
  else if(selecting && event == EVENT_LBUTTONUP) {
    end_point = Point(x, y);
    selecting = false;
    roiImage  = (*((Mat*)userdata))(box_selecting);
    findHoles();
  }
  // 选框已启用 + 鼠标移动
  else if(selecting && event == EVENT_MOUSEMOVE) {
    Mat tempImage = (*((Mat*)userdata)).clone();
    rectangle(tempImage, box_selecting, Scalar(0, 0, 255), 13);
    imshow("holedetect", tempImage);
  }
}

vector<Vec6d> HoleDetector::detectEllipse(const cv::Mat &grey, const float &scale) {
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
  vector<cv::Vec4f> lines;
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
  Mat holeImage = pImage.clone();
  vector<Point> center_list;
  // 圆形检测
  vector<Vec6d> ellipses = detectEllipse(roiImage, 1.0);
  // 绘制圆形 
  for(size_t i = 0; i < ellipses.size(); i++) {
    Point center((int)ellipses[i][0], (int)ellipses[i][1]);
    Size axes((int)ellipses[i][2] + (int)ellipses[i][3], (int)ellipses[i][2] + (int)ellipses[i][4]);
    double angle(ellipses[i][5]);
    Scalar color = ellipses[i][2] == 0 ? Scalar(255, 255, 0) : Scalar(0, 255, 0);
    center_list.push_back(Point2d(ellipses[i][0], ellipses[i][1]));
    // 在图像上绘制圆心
    circle(holeImage, center, 5, Scalar(0, 0, 255), -1);
    // 在图像上绘制圆，或椭圆形
    ellipse(holeImage, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
  }
  // 判断同心圆
  bool flag = areCirclesApproximatelyConcentric(center_list, 2.0);
  if(flag) {

  }
  // 储存孔洞图像
  Mat resizedImage;
  // 计算缩放比例
  int targetWidth = params.size_holeImage;
  float scale = static_cast<float>(targetWidth) / holeImage.cols;
  // 计算缩放后的高度
  int targetHeight = static_cast<int>(holeImage.rows * scale);
  // 缩放图片
  resize(holeImage, resizedImage, cv::Size(targetWidth, targetHeight));
  holelist_image.push_back(resizedImage);
}

bool HoleDetector::areCirclesApproximatelyConcentric(const vector<Point2d> &circles, float deviationThreshold) {
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
  
  
  Point resulCenter = startp + (cv::Point)averageCenter;
  if (deviation > deviationThreshold) {
    cout << "错误：找到的圆没有近似同心！" << endl;
    return false
  }
  else {
    cout << "所有圆心偏差小于阈值，近似同心，求得圆心为：" << resulCenter << endl;
    holelist.push_back(resulCenter);
    return true;
  }

  return; 
}