/*
 * @Author: AIkikaze wenwenziy@163.com
 * @Date: 2023-05-15 15:17:56
 * @LastEditors: Alkikaze wemwemziy@163.com
 * @LastEditTime: 2023-07-10 08:20:46
 * @FilePath: \Cplusplus-playground\Projects_cpp\project_colorSeg+holeDetect\cpp\main.cpp
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
// 创建颜色分割对象
ColorSegment cs;
// 创建圆孔检测对象
HoleDetector hd_model, hd_target;
// 计算结果
vector<String> name_list;
vector<double> result_list;

void colorbyParts(const Mat &src) {
  // 初始化
  cs.setImage(src);

  // 图像处理
  cs.processImage();

  // 色彩检查
  // cs.colorCheck();

  // 添加颜色范围
  cs.addColorRange("green", Scalar(55, 90, 60), Scalar(65, 225, 200));
  cs.addColorRange("red", Scalar(174, 150, 150), Scalar(6, 220, 200));
  cs.addColorRange("light brown", Scalar(18, 90, 150), Scalar(22, 225, 200));
  cs.addColorRange("yellow", Scalar(28, 150, 150), Scalar(32, 220, 200));
  cs.addColorRange("cyan", Scalar(85, 150, 150), Scalar(95, 225, 225));
  cs.addColorRange("dark blue", Scalar(115, 150, 90), Scalar(125, 225, 200));
  cs.addColorRange("dark brown", Scalar(14, 150, 90), Scalar(18, 225, 200));
  cs.addColorRange("purple", Scalar(140, 150, 90), Scalar(150, 225, 200));
  cs.addColorRange("pink", Scalar(155, 150, 90), Scalar(170, 225, 200));

  // 绘制颜色分割区域
  cs.createColorSegment();

  // 显示颜色蒙版
  // cs.showColorMasks();
}

void holesMatch(const Mat &model_image, const Mat &target_image) {
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

  waitKey();
  destroyWindow("holes in modelImage");
  destroyWindow("holes in targetImage");

  // 进行图像配准计算
  vector<Mat> masks;
  cs.copyTo(name_list, masks);
  Mat resultImage = target_image.clone();
  for(const auto &mask : masks) {
    // 进行仿射变换
    Mat warpedImage;
    warpAffine(mask, warpedImage, affineMatrix, target_image.size());

    int tol_num = 0;
    int target_num = 0;
   
    for(int i = 0; i < warpedImage.rows; i++) {
      for(int j = 0; j < warpedImage.cols; j++) { 
        if(!warpedImage.at<uchar>(i, j)) {
          continue;
        }
        else {
          tol_num++;
          if(resultImage.at<Vec3b>(i, j)[0] > 128) {
            target_num++;
            resultImage.at<Vec3b>(i, j) = model_image.at<Vec3b>(i, j);
          }
        }
      }
    }
    result_list.push_back((double)target_num / (double)tol_num);

    // // 显示图像
    // namedWindow("resultImage", WINDOW_NORMAL);
    // imshow("resultImage", resultImage);

    // // 等待键入
    // waitKey();
  }
  // destroyAllWindows();
  imwrite("../imagelib/resultImage.jpg", resultImage);
}

int main() {
  // 读取图像
  Mat model = imread("../imagelib/model_1.jpg", IMREAD_COLOR);
  Mat target = imread("../imagelib/test.TIFF", IMREAD_COLOR);

  // 颜色分割
  colorbyParts(model);

  // 通过圆孔定位进行图像配准
  holesMatch(model, target);

  // 输出结果
  cout << "---------- 输出计算结果 ----------" << endl;
  for(size_t i = 0; i < result_list.size(); i++) {
    cout << "ratio of Area in color " << name_list[i] << ":  " << result_list[i] << endl;
  }
  cin.get();

  return 0;
}
