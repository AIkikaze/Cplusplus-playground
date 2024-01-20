#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

template <class Number>
struct even : public unary_function<Number, bool> {
  bool operator()(Number x) { return (x & 1) == 0; }
};

template <class AdaptablePredicate>
class my_unary_negate {
 private:
  AdaptablePredicate pred;
 public:
  using argument_type = typename AdaptablePredicate::argument_type;
  using result_type = typename AdaptablePredicate::result_type;

  my_unary_negate(const AdaptablePredicate& p) : pred(p) {}
  bool operator()(const argument_type& x) { return !pred(x); }
};

template <class Number>
inline even<Number>::result_type odd(Number x) {
  return my_unary_negate(even<Number>{})(x);
}

int main() {
  vector<int> v = { 1, 3, 2, 7, 0, 9 };
  auto res_even = find_if(v.begin(), v.end(), even<int>());
  auto res_odd = find_if(v.begin(), v.end(), odd<int>);
  cout << "the even result is " << *res_even << endl;
  cout << "the odd result is " << *res_odd << endl;
  system("pause");
  return 0;
}