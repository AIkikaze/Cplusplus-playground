#include "line2d.hpp"
using namespace cv;
using namespace std;
using namespace line2d;
extern double __time__relocate__;
extern double __time__produceroi__;

void template_test() {
    Mat image = imread("../imagelib/mount.png", IMREAD_COLOR);

    Ptr<shapeInfo_producer> sip = shapeInfo_producer::load_config(image);
    sip->scale_range = {0.8, 1.2};
    sip->scale_step = 0.1;
    sip->produce_infos();

    Template::TemplateParams params;
    params.num_features = 50;
    Ptr<Template> tp = makePtr<Template>(params);

    for (const auto &info : sip->Infos_constptr()) {
        cout << "angle: " << info.angle << endl;
        cout << "scale: " << info.scale << endl;

        Mat templateImage;
        sip->src_of(info).copyTo(templateImage, sip->mask_of(info));

        if (!tp->iscreated())
            tp->create_from(templateImage);
        vector<Template::Feature> featurePoints = tp->relocate_by(info);

        Point center(templateImage.cols / 2, templateImage.rows / 2);
        for (const auto &p : featurePoints) {
            circle(templateImage, center + p.p_xy, 1, Scalar(0, 0, 255), -1);
        }

        namedWindow("templateImage", WINDOW_NORMAL);
        imshow("templateImage", templateImage);
        waitKey();
    }
}

int main() {
    // template_test();

    Mat sourceImage = imread("../imagelib/source_2.bmp", IMREAD_COLOR);
    Mat templateImage = imread("../imagelib/template_2.bmp", IMREAD_COLOR);

    Ptr<shapeInfo_producer> sip = makePtr<shapeInfo_producer>(templateImage);
    sip->angle_range = {0, 360};
    sip->angle_step = 1.0;
    sip->scale_range = {0.5, 2.0};
    sip->scale_step = 0.1;
    sip->produce_infos();

    Template::TemplateParams params;
    params.num_features = 400;
    params.nms_kernel_size = 5;
    params.scatter_distance = 12.0f;

    Timer time;
    Detector detector;
    detector.match(sourceImage, templateImage, 70, params);
    time.out("模板匹配运行完毕!");
    cout << "特征点旋转运算时间: " << __time__relocate__ << endl;
    cout << "联通域分析运算时间: " << __time__produceroi__ << endl;

    detector.draw();

    return 0;
}
