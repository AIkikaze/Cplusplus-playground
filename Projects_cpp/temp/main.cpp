#include "header.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;
#define eps 1e-5

template <typename T>
auto access(T &mat, int y, int x) {
}

int main() {
  Mat A;
  A.create(Size(3, 3), CV_8U);

  cout << "rows: " << A.rows << endl;

  cout << "cols: " << A.cols << endl;

  cout << "channals: " << A.channels() << endl;

  cout << "step1: " << A.step1() << endl;

  cout << "step: " << A.step << endl;

  for (int r = -1; r < 4; r++) {
    for (int l = -1; l < 4; l++) {
      cout << "At (" << r << "," << l << ")" << endl;

      cout << "A.at: " << (int)A.at<uchar>(r, l) << endl;

      uchar &cur = A.at<uchar>(r, l);
      cout << "&A.at: " << (int)cur << endl;

      if (&A.at<uchar>(r, l) == nullptr)
        cout << "null" << endl;
      else 
        cout << &A.at<uchar>(r, l) << endl;
    }
  }
  cin.get();
  return 0;
}