#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;

class line_iterator {
  istream* in;
  string line;
  bool is_valid;
  void read() {
    if(*in)
      getline(*in, line);
    is_valid = (*in) ? true : false;
  }

 public:
  using iterator_category = input_iterator_tag;
  using value_type = string;
  using pointer = const string*;
  using reference = const string&;
  using difference_type = ptrdiff_t;

  line_iterator() : in(&cin), is_valid(false) {}
  line_iterator(istream& s) : in(&s) { read(); }

  reference operator*() const { return line; }
  pointer operator->() const { return &line; }
  line_iterator operator++() {
    read();
    return *this;
  }
  line_iterator operator++(int) {
    line_iterator tmp = *this;
    read();
    return tmp;
  }
  bool operator==(const line_iterator& i) const {
    return (in == i.in && is_valid == i.is_valid) ||
          (is_valid == false && i.is_valid == false);
  }
  bool operator!=(const line_iterator& i) const {
    return !(*this == i);
  }
};

int main() {
  line_iterator iter(cin);
  line_iterator end_of_file;
  vector<string> v(iter, end_of_file);
  sort(v.begin(), v.end());
  copy(v.begin(), v.end(), ostream_iterator<string>(cout, "\n"));
  system("pause");
  return 0;
}

