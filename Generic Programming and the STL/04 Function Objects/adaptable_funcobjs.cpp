#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/// @brief old method to get argument_type of a unary func
template <class Arg, class Result>
struct my_unary_function {
  using argument_type = Arg;
  using result_type = Result;
};

/// @brief old method to get argument_type of a binary func
template <class Arg1, class Arg2, class Result>
struct my_binary_function {
  using first_argument_type = Arg1;
  using second_argument_type = Arg2;
  using result_type = Result;
};

template <class Number> 
struct even : public my_unary_function<Number, bool> {
  bool operator()(Number x) const { return (x & 1) == 0; }
};

int main() {
  vector<int> v = { 1, 3, 2, 7, 0, 9 };
  auto res = find_if(v.begin(), v.end(), even<int>());
  cout << "the result is " << *res << endl;
  system("pause");
  return 0;
}