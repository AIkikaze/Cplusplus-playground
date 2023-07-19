#include <algorithm>
#include <iostream>
#include <iterator>
using namespace std;

template <typename T>
class MyRandomAccessIterator {
 public:
  using self = MyRandomAccessIterator;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;

  explicit MyRandomAccessIterator(pointer ptr = nullptr) : ptr(ptr) {}

  reference operator*() const { return *ptr; }
  pointer operator->() const { return ptr; }

  self& operator++() { ++ptr; return *this; }
  self operator++(int) { self temp = *this; ++ptr; return temp; }

  self& operator--() { --ptr; return *this; }
  self operator--(int) { self temp = *this; --ptr; return temp; }

  self operator+(difference_type n) const { return self(ptr + n); }
  friend self operator+(difference_type n, const self& rhs) { return self(rhs.ptr + n); }
  self& operator+=(difference_type n) { ptr += n; return *this; }
  self operator-(difference_type n) const { return self(ptr - n); }
  friend self operator-(difference_type n, const self& rhs) { return self(rhs.ptr - n); }
  self& operator-=(difference_type n) { ptr -= n; return *this; } 

  difference_type operator-(const self& other) const { return ptr - other.ptr; }

  reference operator[](difference_type n) const { return *(ptr + n); }

  bool operator==(const self& other) const { return ptr == other.ptr; }
  bool operator!=(const self& other) const { return ptr != other.ptr; }

  bool operator<(const self& other) const { return ptr < other.ptr; }

  // 小于等于比较
  bool operator<=(const self& other) const { return ptr <= other.ptr; }

  // 大于比较
  bool operator>(const self& other) const { return ptr > other.ptr; }

  // 大于等于比较
  bool operator>=(const self& other) const { return ptr >= other.ptr; }

 private:
  pointer ptr;
};

int main() {
  int arr[] = { 5, 7, 1, 9, 2 };

  MyRandomAccessIterator<int> begin(arr);
  MyRandomAccessIterator<int> end(arr + 5);

  // 使用自定义迭代器遍历数组并输出元素
  cout << "Unsorted arr: ";
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  sort(begin, end);

  // 使用自定义迭代器遍历数组并输出元素
  cout << "Sorted arr: ";
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  // 使用自定义迭代器遍历 vector 并输出元素
  cout << "Using MyRandomAccessIterator to iterate through the vector: ";
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  // 使用自定义迭代器反向遍历 vector 并输出元素
  cout << "Using MyRandomAccessIterator to iterate the vector in reverse: ";
  for (auto it = begin; it != end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  // 测试自定义迭代器的算术运算符重载
  auto it = MyRandomAccessIterator<int>(arr);
  it += 2;
  cout << "Element at index 2: " << *it << endl;  // 输出：Element at index 2: 3

  it -= 1;
  cout << "Element at index 1: " << *it << endl;  // 输出：Element at index 1: 2

  auto it2 = it + 2;
  cout << "Element at index 3: " << *it2 << endl;  // 输出：Element at index 3: 4

  auto it3 = it2 - 1;
  cout << "Element at index 2: " << *it3 << endl;  // 输出：Element at index 2: 3

  cout << "&it: " << &*it << " ";
  cout << "&it2: " << &*it2 << " ";
  cout << "&it3: " << &*it3 << " ";
  cout << endl;

  // 测试自定义迭代器的比较运算符重载
  if (it < it2) {
    cout << "it < it2" << endl;  // 输出：it < it2
  }

  if (it2 > it) {
    cout << "it2 > it" << endl;  // 输出：it2 > it
  }

  if (it <= it3) {
    cout << "it <= it3" << endl;  // 输出：it <= it3
  }

  if (it3 >= it) {
    cout << "it3 >= it" << endl;  // 输出：it3 >= it
  }

  system("pause");

  return 0;
}