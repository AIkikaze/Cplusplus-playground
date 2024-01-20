#include <iostream>
#include <algorithm>
#include <bits/c++config.h>
using namespace std;

template <class T, size_t N>
struct block {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  using difference_type = std::ptrdiff_t;
  using size_type = size_t;

  typedef pointer iterator;
  typedef const_pointer const_iterator;

  iterator begin() { return data; }
  iterator end() { return data + N; }

  const_iterator begin() const { return data; }
  const_iterator end() const { return data + N; }

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  reference operator[](size_type n) { return data[n]; }
  const_reference operator[](size_type n) const { return data[n]; }

  size_type size() const { return N; }
  size_type max_size() const { return N; }
  bool empty() const { return N == 0; }

  T data[N];
};

int main() {
  block<int, 10> A = { 1, 8, 2, 5, 4, 3, 9, 6, 7, 0 };

  for (auto &val : A) {
    cout << val << " ";
  }
  cout << endl;

  sort(A.begin(), A.end());

  for (auto &val : A) {
    cout << val << " ";
  }
  cout << endl;

  system("pause");

  return 0;
}