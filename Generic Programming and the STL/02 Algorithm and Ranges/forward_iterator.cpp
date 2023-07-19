#include <iostream>
using namespace std;

// 自定义 forward iterator
template <typename T>
class MyForwardIterator {
 private:
  T *ptr;

 public:
  using self = MyForwardIterator&;
  using reference = T&;
  using pointer = T*;

  // 构造函数
  MyForwardIterator(T *p) : ptr(p) {}

  // 解引用操作符
  reference operator*() const { return *ptr; }
  pointer operator->() const { return *ptr; }

  // 前置递增操作符
  self operator++() { ++ptr; return *this; }

  // 后置递增操作符
  self operator++(int) { auto tmp = *this; ++ptr; return tmp; }

  // 比较操作符
  bool operator==(const MyForwardIterator &other) const {
    return ptr == other.ptr;
  }

  bool operator!=(const MyForwardIterator &other) const {
    return ptr != other.ptr;
  }
};

int main() {
  int arr[] = {1, 2, 3, 4, 5};
  MyForwardIterator<int> begin(arr);
  MyForwardIterator<int> end(&arr[5]);

  // 使用自定义迭代器遍历数组并输出元素
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  // 使用自定义迭代器遍历数组并输出元素
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  // 使用STL的replace函数，将数组中的元素2替换为100
  replace(begin, end, 2, 100);

  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  return 0;
}
