#include <iostream>
#include <vector>
#include <list>
#include <set>
using namespace std;

template <class Iterator>
struct my_iterator_traits {
  typedef typename Iterator::value_type value_type;
  typedef typename Iterator::reference reference;
  typedef typename Iterator::pointer pointer;
  typedef typename Iterator::difference_type difference_type;
  typedef typename Iterator::iterator_category iterator_category;
};

template <typename T>
void type_check(const T& variable) {
  cout << "Variable type: " << typeid(variable).name() << endl;
  cout << "value_type: " << typeid(typename my_iterator_traits<T>::value_type).name() << endl;
  cout << "reference: " << typeid(typename my_iterator_traits<T>::reference).name() << endl;
  cout << "pointer: " << typeid(typename my_iterator_traits<T>::pointer).name() << endl;
  cout << "difference_type: " << typeid(typename my_iterator_traits<T>::difference_type).name() << endl;
  cout << "iterator_category: " << typeid(typename my_iterator_traits<T>::iterator_category).name() << endl;
}


int main() {
  set<double> arr = { 1, 2, 3 };
  auto begin = arr.begin();

  type_check(begin);

  system("pause");

  return 0;
}