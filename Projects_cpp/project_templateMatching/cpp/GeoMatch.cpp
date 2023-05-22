#include "GeoMatch.hpp"
using namespace std;
using namespace cv;

GeoMatch::GeoMatch() { modelDefined = false; }

void GeoMatch::setSourceImage(const cv::Mat &sImage) {
  sourceImage = sImage.clone();
}

void GeoMatch::setTempImage(const cv::Mat &tImage) {
  tempImage = tImage.clone();
}

void GeoMatch::processImage() {
  // 将待匹配图片和模板图片转化为灰度图
  if (sourceImage.type() == CV_8UC3)
    cvtColor(sourceImage, sourceImage, COLOR_BGR2GRAY);
  if (tempImage.type() == CV_8UC3)
    cvtColor(tempImage, tempImage, COLOR_BGR2GRAY);
}

void GeoMatch::createGeoMatchModel(double maxContrast,
                                   double minContrast) {
  CV_Assert(!tempImage.empty());

  /// @variables:
  float maxGrad = -1e9f;
  Mat edges = Mat::zeros(tempImage.rows, tempImage.cols, CV_32F);
  Mat angles = Mat::zeros(tempImage.rows, tempImage.cols, CV_8U);
  Mat gx = Mat::zeros(tempImage.rows, tempImage.cols, CV_32F);
  Mat gy = Mat::zeros(tempImage.rows, tempImage.cols, CV_32F);

  // 使用 Sobel 算子计算 x,y 方向上的一阶差分矩阵
  Sobel(tempImage, gx, CV_32F, 1, 0, 3);
  Sobel(tempImage, gy, CV_32F, 0, 1, 3);

  // 进行梯度和方向角的计算
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      // 计算梯度模长
      edges.at<float>(i, j) = sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                                   gy.at<float>(i, j) * gy.at<float>(i, j));

      // 记录梯度最大值，以便后续进行归一化
      if (edges.at<float>(i, j) > maxGrad) maxGrad = edges.at<float>(i, j);

      // 计算方向角
      double angle = fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));

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
  copyMakeBorder(edges, edges, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(0));
  for (int i = 1; i < edges.rows - 1; i++) {
    for (int j = 1; j < edges.cols - 1; j++) {
      float prevPixel, nextPixel;
      switch ((int)angles.at<uchar>(i - 1, j - 1)) {
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
        edges.at<float>(i, j) = 0;
      // 在这里顺带完成梯度矩阵的归一化
      else
        edges.at<float>(i, j) = 255.0f * (edges.at<float>(i, j) / maxGrad);
    }
  }
  // 裁取被扩张的边界
  edges = edges(Range(1, edges.rows - 1), Range(1, edges.cols - 1));
  // 将 edges 转化为 8bit 灰度图
  edges.convertTo(edges, CV_8U);

  // 使用 Hysteresis 方法将 edges 二值化
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<uchar>(i, j) < minContrast) edges.at<uchar>(i, j) = 0;
      if (edges.at<uchar>(i, j) > maxContrast) edges.at<uchar>(i, j) = 255;
    }
  }
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<uchar>(i, j) < minContrast ||
          edges.at<uchar>(i, j) > maxContrast)
        continue;
      for (int di = -1; di < 2; di++) {
        for (int dj = -1; dj < 2; dj++) {
          if (edges.at<uchar>(i + di, j + dj) == 255) edges.at<uchar>(i, j) = 255;
        }
      }
      if (edges.at<uchar>(i, j) != 255) edges.at<uchar>(i, j) = 0;
    }
  }

  // 将 edges 矩阵中有效像素对应的坐标，以及 gx, gy 存入梯度序列
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (!edges.at<uchar>(i, j)) continue;
      gradVecList.push_back(
          coorGradient(Point(i, j), Vec2f(gx.at<float>(i, j))));
    }
  }

  // 记录 modeldefined
  if (!gradVecList.empty()) modelDefined = true;
}

void GeoMatch::createGeoMatchModel(const cv::Mat &tImage, double maxContrast,
                                   double minContrast) {
  /// @variables:
  float maxGrad = -1e9f;
  Mat edges = Mat::zeros(tImage.rows, tImage.cols, CV_32F);
  Mat angles = Mat::zeros(tImage.rows, tImage.cols, CV_8U);
  Mat gx = Mat::zeros(tImage.rows, tImage.cols, CV_32F);
  Mat gy = Mat::zeros(tImage.rows, tImage.cols, CV_32F);

  // 载入模板图片
  setTempImage(tImage);

  // 使用 Sobel 算子计算 x,y 方向上的一阶差分矩阵
  Sobel(tempImage, gx, CV_32F, 1, 0, 3);
  Sobel(tempImage, gy, CV_32F, 0, 1, 3);

  // 进行梯度和方向角的计算
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      // 计算梯度模长
      edges.at<float>(i, j) = sqrt(gx.at<float>(i, j) * gx.at<float>(i, j) +
                                   gy.at<float>(i, j) * gy.at<float>(i, j));

      // 记录梯度最大值，以便后续进行归一化
      if (edges.at<float>(i, j) > maxGrad) maxGrad = edges.at<float>(i, j);

      // 计算方向角
      double angle = fastAtan2(gy.at<float>(i, j), gx.at<float>(i, j));

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
  copyMakeBorder(edges, edges, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(0));
  for (int i = 1; i < edges.rows - 1; i++) {
    for (int j = 1; j < edges.cols - 1; j++) {
      float prevPixel, nextPixel;
      switch ((int)angles.at<uchar>(i - 1, j - 1)) {
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
        edges.at<float>(i, j) = 0;
      // 在这里顺带完成梯度矩阵的归一化
      else
        edges.at<float>(i, j) = 255.0f * (edges.at<float>(i, j) / maxGrad);
    }
  }
  // 裁取被扩张的边界
  edges = edges(Range(1, edges.rows - 1), Range(1, edges.cols - 1));
  // 将 edges 转化为 8bit 灰度图
  edges.convertTo(edges, CV_8U);

  // 使用 Hysteresis 方法将 edges 二值化
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<uchar>(i, j) < minContrast) edges.at<uchar>(i, j) = 0;
      if (edges.at<uchar>(i, j) > maxContrast) edges.at<uchar>(i, j) = 255;
    }
  }
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (edges.at<uchar>(i, j) < minContrast ||
          edges.at<uchar>(i, j) > maxContrast)
        continue;
      for (int di = -1; di < 2; di++) {
        for (int dj = -1; dj < 2; dj++) {
          if (edges.at<uchar>(i + di, j + dj) == 255) edges.at<uchar>(i, j) = 255;
        }
      }
      if (edges.at<uchar>(i, j) != 255) edges.at<uchar>(i, j) = 0;
    }
  }

  // 将 edges 矩阵中有效像素对应的坐标，以及 gx, gy 存入梯度序列
  for (int i = 0; i < edges.rows; i++) {
    for (int j = 0; j < edges.cols; j++) {
      if (!edges.at<uchar>(i, j)) continue;
      gradVecList.push_back(
          coorGradient(Point(i, j), Vec2f(gx.at<float>(i, j))));
    }
  }

  // 记录 modeldefined
  if (!gradVecList.empty()) modelDefined = true;
}

Mat GeoMatch::getScoreMap() {
  /// @variables:
  Mat Gx = Mat::zeros(sourceImage.rows, sourceImage.cols, CV_32F);
  Mat Gy = Mat::zeros(sourceImage.rows, sourceImage.cols, CV_32F);
  Mat Sm = Mat::zeros(sourceImage.rows, sourceImage.cols, CV_32F);

  if (!modelDefined) {
    cout << "错误：模板未定义！" << endl;
    return Sm;
  }

  // Sobel 计算源图像的一阶差分
  Sobel(sourceImage, Gx, CV_32F, 1, 0, 3);
  Sobel(sourceImage, Gy, CV_32F, 0, 1, 3);

  /*
  计算相似度，相似度公式为：
  Scores_map (u, v) = 1 / m \sum_{i=1}^m { g_i * G(u+x_i, v+y_i) }
                                        / { |g_i| * |G(u+x_i, v+y_i)| }
  */
  for (int i = 0; i < sourceImage.rows; i++) {
    for (int j = 0; j < sourceImage.cols; j++) {
      for (const auto &g : gradVecList) {
        int u, v;
        float _gradTemp, _gradSource;
        u = i + g.coordiante.y;
        v = j + g.coordiante.x;

        if (u < 0 || v < 0 || u > sourceImage.rows - 1 ||
            v > sourceImage.cols - 1)
          continue;

        if (sqrt(Gx.at<float>(u, v) * Gx.at<float>(u, v) + Gy.at<float>(u, v) * Gy.at<float>(u, v)) == 0)
          _gradSource = 0.0f;
        else
          _gradSource = 1.0f / sqrt(Gx.at<float>(u, v) * Gx.at<float>(u, v) + Gy.at<float>(u, v) * Gy.at<float>(u, v));
        _gradTemp = 1.0f / sqrt(g.edgesXY[0] * g.edgesXY[0] +
                                g.edgesXY[1] * g.edgesXY[1]);

        Sm.at<float>(i, j) +=
            g.edgesXY[0] * Gx.at<float>(u, v) + g.edgesXY[1] * Gy.at<float>(u, v) * _gradTemp *
            _gradSource;
      }
      Sm.at<float>(i, j) /= gradVecList.size();
      cout << Sm.at<float>(i, j) << endl;
      cin.get();
   }
  }

  normalize(Sm, Sm, 0, 255, NORM_MINMAX);
  Sm.convertTo(Sm, CV_8U);

  return Sm;
}

float GeoMatch::findGeoMatchModel(float minScore = 0.0f,
                                  float greediness = 0.0f) {
  if (!modelDefined) {
    cout << "错误：模板未定义！" << endl;
    return -1.0f;
  }

  /// @variables:
  int sumOfCoords = 0;
  float resultScorce = 0.0;
  float partialSum = 0.0;
  float partialScore = 0.0;
  float normMinScore = minScore / gradVecList.size();
  float normGreediness =
      greediness < 1.0f ? ((1 - greediness * minScore) / (1 - greediness)) /
                              gradVecList.size()
                        : 1.0 / gradVecList.size();
  Mat Gx = Mat::zeros(sourceImage.rows, sourceImage.cols, CV_32F);
  Mat Gy = Mat::zeros(sourceImage.rows, sourceImage.cols, CV_32F);

  // Sobel 计算源图像的一阶差分
  Sobel(sourceImage, Gx, 1, 0, 3);
  Sobel(sourceImage, Gy, 0, 1, 3);

  /*
  计算相似度，相似度公式为：
  Scores_map (u, v) = 1 / m \sum_{i=1}^m { g_i * G(u+x_i, v+y_i) }
                                        / { |g_i| * |G(u+x_i, v+y_i)| }
  */
  for (int i = 0; i < sourceImage.rows; i++) {
    for (int j = 0; j < sourceImage.cols; j++) {
      partialSum = 0.0;
      sumOfCoords = 0;
      for (const auto &g : gradVecList) {
        int u, v;
        float _gradTemp, _gradSource;
        u = i + g.coordiante.y;
        v = j + g.coordiante.x;
        sumOfCoords++;

        if (u < 0 || v < 0 || u > sourceImage.rows - 1 ||
            v > sourceImage.cols - 1)
          continue;

        if (sqrt(Gx.at<float>(u, v) * Gx.at<float>(u, v) + Gy.at<float>(u, v) * Gy.at<float>(u, v)) == 0)
          _gradSource = 0.0f;
        else
          _gradSource = 1.0f / sqrt(Gx.at<float>(u, v) * Gx.at<float>(u, v) + Gy.at<float>(u, v) * Gy.at<float>(u, v));
        _gradTemp = 1.0f / sqrt(g.edgesXY[0] * g.edgesXY[0] +
                                g.edgesXY[1] * g.edgesXY[1]);

        partialSum +=
            g.edgesXY[0] * Gx.at<float>(u, v) + g.edgesXY[1] * Gy.at<float>(u, v) * _gradTemp *
            _gradSource;

        partialScore = partialSum / sumOfCoords;

        if (partialScore > min((minScore - 1) + normGreediness * sumOfCoords,
                               normMinScore * sumOfCoords))
          ;
        break;
      }

      if (partialScore > resultScorce) {
        cout << "Partial Score is :" << partialScore << endl;
        resultScorce = partialScore;
      }
    }
  }

  return resultScorce;
}

void GeoMatch::show() {
  namedWindow("sourceImage");
  namedWindow("tempImage");
  imshow("sourceImage", sourceImage);
  imshow("tempImage", tempImage);
  waitKey();
  destroyWindow("sourceImage");
  destroyWindow("tempImage");
}
