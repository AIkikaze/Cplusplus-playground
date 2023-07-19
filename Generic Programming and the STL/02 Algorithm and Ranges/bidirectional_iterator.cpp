#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 自定义 forward iterator
template <typename T>
class MyBidirectionalIterator {
 private:
  T *ptr;

 public:
  using self = MyBidirectionalIterator;
  using value_type = T;
  using reference = T&;
  using pointer = T*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;

  // 构造函数
  MyBidirectionalIterator(T *p = nullptr) : ptr(p) {}

  // 解引用操作符
  reference operator*() const { return *ptr; }
  pointer operator->() const { return *ptr; }

  // 前置递增操作符
  self& operator++() { ++ptr; return *this; }

  // 后置递增操作符
  self operator++(int) { auto tmp = *this; ++ptr; return tmp; }

  // 前置递增操作符
  self& operator--() { --ptr; return *this; }

  // 后置递增操作符
  self operator--(int) { auto tmp = *this; --ptr; return tmp; }

  // 比较操作符
  bool operator==(const self &other) const {
    return ptr == other.ptr;
  }

  bool operator!=(const self &other) const {
    return ptr != other.ptr;
  }
};

int main() {
  int arr[] = { 1, 2, 3, 4, 5 };
  MyBidirectionalIterator<int> begin(arr);
  MyBidirectionalIterator<int> end(&arr[5]);

  vector<int> v;
  v.resize(5);
  reverse_copy(begin, end, v.begin());

  for (auto val : v) {
    cout << val << " ";
  }
  cout << endl;

  system("pause");

  return 0;
}
