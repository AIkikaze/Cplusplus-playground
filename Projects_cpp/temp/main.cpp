#include <cstdlib>  // For random number generation
#include <iostream>
#include <vector>

// 2D 点结构体
struct Point {
  int x;
  int y;

  Point(int x, int y) : x(x), y(y) {}
};

// 3D 点结构体
struct Point3D : Point {
  int z;

  Point3D(int x, int y, int z) : Point(x, y), z(z) {}
};

// 点工厂
template <typename PointType>
class PointFactory {
 public:
  int numberOfPoints;

  PointFactory(int x) : numberOfPoints(x) {}

  std::vector<PointType> createPoints() {
    std::vector<PointType> points;
    for (int i = 0; i < numberOfPoints; i++) {
      if constexpr (std::is_same<PointType, Point>::value) {
        points.emplace_back(rand() % 10, rand() % 10);
      } else if constexpr (std::is_same<PointType, Point3D>::value) {
        points.emplace_back(rand() % 10, rand() % 10, rand() % 10);
      } else {
        throw std::invalid_argument("Unsupported PointType");
      }
    }
    return points;
  }
};

int main() {
  srand(time(NULL));  // Seed the random number generator

  PointFactory<Point> pointFactory(5);
  for (const auto& point : pointFactory.createPoints()) {
    std::cout << "2D Point: (" << point.x << ", " << point.y << ")"
              << std::endl;
  }

  PointFactory<Point3D> point3DFactory(5);
  for (const auto& point3D : point3DFactory.createPoints()) {
    std::cout << "3D Point: (" << point3D.x << ", " << point3D.y << ", "
              << point3D.z << ")" << std::endl;
  }

  system("pause");
  return 0;
}