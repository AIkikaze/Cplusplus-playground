#include <iostream>
#include <vector>
#include <type_traits>
using namespace std;

template <class InputIterator, class Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred) {
  while (first != last && !pred(*first))
    ++first;
  return first;
}

bool is_even(int x) { return (x & 1) == 0; }

template <class Number>
struct even {
  bool operator()(Number x) const { return (x & 1) == 0; }
};

template <class Number>
struct is_number {
  Number val;
  is_number(Number x = Number{}) : val(x) {}
  bool operator()(Number x) { return val == x; }
};

template <class InputIterator, class Predicate>
void show_ressult(InputIterator first, InputIterator last, Predicate pred) {
  auto res = find_if(first, last, pred);
  if (res != last) {
    cout << "the found result is " << *res << " by " << typeid(pred).name() << endl;
  } else
    cout << "nothing found !" << endl;
}

int main() {
  vector<int> v = { 1, 3, 7, 8, 2 };

  show_ressult(v.begin(), v.end(), is_even);
  show_ressult(v.begin(), v.end(), even<int>());
  show_ressult(v.begin(), v.end(), is_number(8));

  system("pause");
  
  return 0;
}
