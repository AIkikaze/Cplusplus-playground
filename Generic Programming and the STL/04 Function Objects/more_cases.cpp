#include <iostream>
#include <concepts>
#include <vector>
using namespace std;

template <forward_iterator ForwardIterator, class BinaryPredicate>
ForwardIterator adjacent_find(ForwardIterator first, ForwardIterator last, BinaryPredicate pred) {
  if (first == last) return last;
  ForwardIterator next = first;
  while (++next != last) {
    if (pred(*first, *next))
      return first;
    first = next;
  }
  return last;
}

template <input_iterator InputIterator, class OutputIterator, class UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction f) {
  while (first != last) *result++ = f(*first++);
  return result;
}

bool my_equal(int x, int y) { return x == y; }

int opposite(int x) { return -x; }

int main() {
  vector<int> v = { 2, 3, 1, 1, 6, 9, 3 };

  auto res = adjacent_find(v.begin(), v.end(), my_equal);
  cout << "the result is " << *res << " = " << *(++res) << endl;

  vector<int> w(10);
  auto new_end = transform(v.begin(), v.end(), w.begin(), opposite);
  for (auto it = w.begin(); it != new_end; ++it) {
    cout << *it << " ";
  }
  cout << endl;

  system("pause");

  return 0;
}