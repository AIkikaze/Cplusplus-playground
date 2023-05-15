#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // 加载待配准的两张图像
    Mat image1 = imread("../imagelib/test.TIFF");
    Mat image2 = imread("../imagelib/model_1.jpg");

    // 显示待配准的两张图像
    namedWindow("test", WINDOW_NORMAL);
    namedWindow("model_1", WINDOW_NORMAL);
    imshow("test", image1);
    imshow("model_1", image2);
    waitKey();

    // // 在图像上手动选择约束点
    // std::vector<Point2f> points1, points2;
    // int numPoints = 4; // 假设选择了4个约束点
    // for (int i = 0; i < numPoints; i++)
    // {
    //     // 在图像1上选择一个约束点
    //     Point2f point1 = Point2f(-1, -1);
    //     while (point1.x == -1 && point1.y == -1)
    //     {
    //         std::cout << "Select a constraint point in Image 1..." << std::endl;
    //         imshow("Image 1", image1);
    //         setMouseCallback("Image 1", [](int event, int x, int y, int flags, void* userdata) {
    //             if (event == EVENT_LBUTTONDOWN)
    //                 *static_cast<Point2f*>(userdata) = Point2f(x, y);
    //         }, &point1);
    //         waitKey(0);
    //     }

    //     // 在图像2上选择相应的约束点
    //     Point2f point2 = Point2f(-1, -1);
    //     while (point2.x == -1 && point2.y == -1)
    //     {
    //         std::cout << "Select the corresponding point in Image 2..." << std::endl;
    //         imshow("Image 2", image2);
    //         setMouseCallback("Image 2", [](int event, int x, int y, int flags, void* userdata) {
    //             if (event == EVENT_LBUTTONDOWN)
    //                 *static_cast<Point2f*>(userdata) = Point2f(x, y);
    //         }, &point2);
    //         waitKey(0);
    //     }

    //     points1.push_back(point1);
    //     points2.push_back(point2);
    // }

    // // 进行透视变换估计
    // Mat homography = findHomography(points1, points2, RANSAC);

    // // 对image1进行透视变换，使其与image2对齐
    // Mat result;
    // warpPerspective(image1, result, homography, image2.size());

    // // 显示配准结果
    // namedWindow("Registered Image 1", WINDOW_NORMAL);
    // imshow("Registered Image 1", result);
    // waitKey(0);

    return 0;
}
