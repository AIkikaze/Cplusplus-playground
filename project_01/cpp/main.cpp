/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-15 15:17:56
 * @LastEditors: AIkikaze wenwenziy@163.com
 * @LastEditTime: 2023-05-16 13:20:47
 * @FilePath: /C++-playground/project_01/cpp/main.cpp
 * @Description:
 *
 */
#include "detectHoles.hpp"
#include "colorSegment.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;

void colorbyParts(const Mat &src) {
  // 创建颜色分割对象
  ColorSegment cs;
  // 初始化
  cs.setImage(src);
  // 图像处理
  cs.processImage();
  // 色彩检查
  cs.colorCheck();
  // 添加颜色范围
  cs.addColorRange("green", Scalar(55, 90, 60), Scalar(65, 220, 140));
  // 绘制颜色分割区域
  cs.createColorSegment();
}

void holesMatch(const Mat &model_image, const Mat &target_image) {
  // 创建圆孔检测对象
  HoleDetector hd_model, hd_target;

  // 默认选取 4 个对应点
  int num_match = 4;
  // 储存定位孔位置
  vector<Point> srcPoints, dstPoints;

  // 对 target 图片进行圆孔定位
  hd_target.setImage(target_image);
  hd_target.processImage();
  while(hd_target.getSize() < num_match)
    hd_target.addNewHole();
  hd_target.drawHoleList("holes in targetImage", 1);
  hd_target.copyHolelistTo(dstPoints);

  // 对 model 图片进行圆孔定位
  hd_model.setImage(model_image);
  hd_model.processImage();
  while(hd_model.getSize() < num_match)
    hd_model.addNewHole();
  hd_model.drawHoleList("holes in modelImage", 1);
  hd_model.copyHolelistTo(srcPoints);

  // 计算仿射变换
  Mat affineMatrix;
  Mat inliers; 
  affineMatrix = estimateAffine2D(srcPoints, dstPoints, inliers);

  // 输出计算结果
  cout << "--------- 仿射变换计算完成 -----------" << endl;
  for(int i = 0; i < num_match; i++) {
    cout << "对应点[" << i+1 << "]:" << srcPoints[i] << " " << dstPoints[i] << endl;
  }
  cout << "求得仿射变换矩阵：" << endl <<  affineMatrix << endl;

  // 进行图像配准
  // 进行仿射变换
  Mat warpedImage;
  warpAffine(model_image, warpedImage, affineMatrix, target_image.size());

  // 色彩叠加
  double alpha = 0.5;  // 图像 model 的权重
  Mat blendedImage;
  addWeighted(warpedImage, alpha, target_image, 1 - alpha, 0, blendedImage);

  // 显示图像
  namedWindow("blendedImage", WINDOW_NORMAL);
  imshow("blendedImage", blendedImage);

  // 等待键入
  waitKey();
  destroyAllWindows();
}

int main() {
  // 读取图像
  Mat model = imread("../imagelib/model_1.jpg", IMREAD_COLOR);
  Mat target = imread("../imagelib/test.TIFF", IMREAD_GRAYSCALE);

  // 颜色分割
  colorbyParts(model);

  // 通过圆孔定位进行图像配准
  // holesMatch(model, target);

  return 0;
}
