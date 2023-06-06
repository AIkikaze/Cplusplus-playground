#include "line2dup.hpp"
using namespace line2Dup;
using namespace std;
using namespace cv;

/// struct Feature

void Feature::read(const FileNode &fn) {
  FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const {
  fs << "[:" << x << y << label << "]";
}

// struct Template

static Rect cropTemplate(vector<Template> &templates) {
  int min_x = numeric_limits<int>::max();
  int max_x = numeric_limits<int>::min();
  int min_y = numeric_limits<int>::max();
  int max_y = numeric_limits<int>::min();
  int highest_level = 0;

  // First: find min/max feature x,y over all pyramid levelds and modalities
  for (int i = 0; i < (int)templates.size(); i++) {
    Template &templ = templates[i];
    for (int j = 0; j < (int)templ.features.size(); j++) {
      // rescaling the position to the lowest level in pyramid
      int x = templ.features[j].x << templ.pyramid_level;
      int y = templ.features[j].y << templ.pyramid_level;
      // comparing
      min_x = min(min_x, x);
      max_x = max(max_x, x);
      min_y = min(min_y, y);
      max_y = max(max_y, y);
      // keep the maximum level
      highest_level = max(highest_level, templ.pyramid_level);
    }
  }

  // Notice: x = ~.x << pyramid_level is always even except pryramid_level == 0
  // and (min_x, min_y) should be the refrence point of the template (_x_, _y_)
  // so we shold better let (min_x, min_y) uniquely corespond to one point in
  // all level
  /// @example: (min_x, min_y) == (11, 5), highest_level == 3
  ///                         at level     0     1     2     3
  /// min_x : 11 -> (11 >> 3) << 3 = 8     8 <-> 4 <-> 2 <-> 1
  /// min_y : 5  -> (5 >> 3) << 3  = 0     0 <-> 0 <-> 0 <-> 0
  min_x = (min_x >> highest_level) << highest_level;
  min_y = (min_y >> highest_level) << highest_level;

  // Second: set width / height and shift all feature positions
  for (int i = 0; i < (int)templates.size(); i++) {
    Template &templ = templates[i];
    templ.width = (max_x - min_x) >> templ.pyramid_level;
    templ.height = (max_x - min_x) >> templ.pyramid_level;
    int offset_x = min_x >> templ.pyramid_level;
    int offset_y = min_y >> templ.pyramid_level;

    for (int j = 0; j < (int)templ.features.size(); j++) {
      /// @todo: why "-=" offset
      templ.features[j].x -= offset_x;
      templ.features[j].y -= offset_y;
    }
  }

  return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Template::read(const FileNode &fn) {
  width = fn["width"];
  height = fn["height"];
  pyramid_level = fn["pyramid_level"];

  FileNode featrues_fn = fn["features"];
  features.resize(featrues_fn.size());
  FileNodeIterator it = featrues_fn.begin(), it_end = featrues_fn.end();
  for (int i = 0; it != it_end; i++, it++) {
    features[i].read(*it);
  }
}

void Template::write(FileStorage &fs) const {
  fs << "width" << width;
  fs << "height" << height;
  fs << "pyramid_level" << pyramid_level;

  fs << "features"
     << "[";
  for (int i = 0; i < (int)features.size(); i++) {
    features[i].write(fs);
  }
  fs << "]";
}

// class Modality

void QuantizedPyramid::selectScatteredFeatures(
    const vector<Candidate> &candidates, vector<Feature> &features,
    size_t num_features, float distance) {
  features.clear();
  float distance_sq = distance * distance;
  int i = 0;

  while (features.size() < num_features) {
    const Candidate &c = candidates[i];
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; j++) {
      Feature &fj = features[j];
      keep =
          (c.f.x - fj.x) * (c.f.x - fj.x) + (c.f.y - fj.y) * (c.f.y - fj.y) >=
          distance_sq;
    }
    if (keep)
      features.push_back(c.f);

    if (++i == (int)candidates.size()) {
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
}

Ptr<Modality> Modality::create(const String &modality_type) {
  if (modality_type == "ColorGradient") 
    return makePtr<ColorGradient>();
  else 
    return Ptr<Modality>();
}

Ptr<Modality> Modality::create(const FileNode &fn) {
  String type = fn["type"];
  Ptr<Modality> modality = create(type);
  modality->read(fn);
  return modality;
}

void line2Dup::colormap(const Mat &quantized, Mat &dst) {
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
    const uchar *quad_r = quantized.ptr(r);
    Vec3b *dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; c++) {
      for (int k = 0; k < 16; k++)
        if (quad_r[c] & (1<<k)) dst_r[c] = colors[k];
    }
  }
}

// #include <opencv2/imgproc.hpp>
void drawFeatures(InputOutputArray img, const vector<Template> &templates, const Point2i &_p_, int size) {
#ifdef OPENCV_IMPROC_HPP
  static Scalar colors[] = {{0, 0, 255}, {0, 255, 0}};
  static int markers[] = {MARKER_SQUARE, MARKER_DIAMOND};

  int modality = 0;
  for (const Template &t : templates) {
    if (t.height != 0) continue;
    for (const Feature &f : t.features) {
      drawMarker(img, _p_ + Point(f.x, f.y), colors[int(modality != 0)], markers[int(modality != 0)], size);
    }
    modality++;
  }
#else
  CV_Error(Error::StsAssert, "need improc module");
#endif
}

/// Color Gradient Modality

class ColorGradientPyramid : public QuantizedPyramid {
public:
  ColorGradientPyramid(
    const Mat &src,
    const Mat &mask,
    float weak_threshold,
    float strong_threshold,
    size_t num_features
    );

  virtual void quantize(Mat &dst) const override;

  virtual bool extractTemplate(Template &templ) const override;

  virtual void pyrDown() override;

protected:
  void update();

  int pyramid_level;
  Mat src;
  Mat mask;

  Mat magnitude;
  Mat angle;

  float weak_threshold;
  float strong_threshold;
  size_t num_features;
}

ColorGradientPyramid::ColorGradientPyramid(
  const Mat &_src,
  const Mat &_mask,
  float _weak_threshold,
  float _strong_threshold,
  size_t _num_features) 
  : src(_src),
    mask(_mask),
    weak_threshold(_weak_threshold),
    strong_threshold(_strong_threshold),
    num_features(_num_features) { update(); }


static void quantizeAngle(
  Mat &magnitude,
  Mat &angle,
  Mat &quantized_angle,
  float threshold) {
  Mat quanized_unfiltered;
  angle.convertTo(quanized_unfiltered, CV_8U, 16.0 / 360.0);

  
}

static void sobelMagnitude(
  const Mat &src,
  Mat &magnitude,
  Mat &sobel_dx,
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
  GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
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

 
}

void ColorGradientPyramid::update() {
  Mat sobel_dx, sobel_dy;
  sobelMagnitude(src, magnitude, sobel_dx, sobel_dy);

  Mat sobel_ag;
  phase(sobel_dx, sobel_dy, sobel_ag, true);

  quantizeAngle(magnitude, sobel_ag, angle, strong_threshold * strong_threshold);
}