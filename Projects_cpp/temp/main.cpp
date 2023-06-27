#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

void PrintHelp() {
    cout << "Usage: program [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -w, --write     Write data to a file" << endl;
    cout << "  -r, --read      Read data from a file" << endl;
    cout << "  -h, --help      Show help message" << endl;
}

struct Arr {
  int cols;
  int rows;
  vector<vector<int>> data;

  friend void operator >>(const cv::FileNode &fn, Arr &arr);
  friend void operator <<(const cv::FileStorage &fs, const Arr &arr);
};

void operator>>(const cv::FileNode& fn, Arr& arr) {
  arr.cols = (int)fn["cols"];
  arr.rows = (int)fn["rows"];
  cv::FileNode fn_vector = fn["vector"];
  arr.data.resize(fn_vector.size());
  auto it = fn_vector.begin(), it_end = fn_vector.end();
  for (int i = 0; it != it_end; it++, i++) {
    for (const auto& value : (*it)) {
      arr.data[i].push_back(static_cast<int>(value));
    }
  }
}

void operator<<(cv::FileStorage& fs, const Arr& arr) {
  fs << "cols" << arr.cols;
  fs << "rows" << arr.rows;
  fs << "vector" << "[";
  for (int i = 0; i < (int)arr.data.size(); i++) {
    fs << arr.data[i];
  }
  fs << "]";
}

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, "{@input||}{help h ||}{write w ||}{read r ||}");
  Arr x;
  x.rows = 4;
  x.cols = 4;
  x.data = {
    {1, 0, 0, 0},
    {2, 0, 1, -1},
    {2, 4, 3, 7},
    {0, -1, 2, 1}
  };

  String filename = parser.get<String>("@input");
  if (parser.has("help")) {
    PrintHelp();
    return 0;
  }
  if (filename.empty()) {
    cerr << "no filename for output" << endl;
    return 1;
  }
  if (parser.has("write")) {
    FileStorage fs(filename, FileStorage::WRITE);
    fs << x;
  }
  if (parser.has("read")) {
    FileStorage fs(filename, FileStorage::READ);
    FileNode rootNode = fs.root(); // 获取根节点
    Arr y;
    rootNode >> y;
 
    cout << "[";
    for (int i = 0; i < (int)y.data.size(); i++) {
      for (auto &value : y.data[i]) {
        cout << " " << value << ",";
      }
      cout << (i == ((int)y.data.size() - 1) ? "]" : "\n");
    }
  }
  return 0;
}