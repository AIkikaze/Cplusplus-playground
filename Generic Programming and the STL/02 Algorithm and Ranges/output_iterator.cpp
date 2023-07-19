#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

template <class T>
class os_iterator {
 private:
  ostream* os;
  const char* string;
 public:
  /// @brief 必要的声明
  using value_type = T;
  using reference = os_iterator<T>&;
  // using pointer = os_iterator<T>*;
  // using difference_type = std::ptrdiff_t;
  // using iterator_category = std::output_iterator_tag;
  
  os_iterator(ostream& s, const char* ch = nullptr) : os(&s), string(ch) {}
  os_iterator(const os_iterator& i) : os(i.os), string(i.string) {}

  os_iterator& operator=(const os_iterator& i) {
    os = i.os;
    string = i.string;
    return *this;
  }
  reference operator=(const T& value) {
    *os << value;
    if (string) *os << string;
    return *this;
  }
  reference operator*() { return *this; }
  reference operator++() { return *this; }
  reference operator++(int) { return *this; }
};

int main() {
  vector<int> v = { 1, 2, 9, 5, 6 };
  copy(v.begin(), v.end(), os_iterator<int>(cout, " "));
  system("pause");
  return 0;
}