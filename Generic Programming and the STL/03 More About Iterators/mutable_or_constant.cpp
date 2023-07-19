#include <vector>
#include <iostream>
using namespace std;

template <typename _T, typename _Ref, typename _Tp>
class iterator_base {
 public:
  using self = iterator_base;
  using value_type = _T;
  using reference = _Ref;
  using pointer = _Tp;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;

  using iterator = iterator_base<_T, _T&, _T*>;
  using const_iterator = iterator_base<_T, const _T&, const _T*>;

  pointer ptr;

  iterator_base(pointer p = nullptr) : ptr(p) {}
  iterator_base(const iterator& x) : ptr(x.ptr) {}

  reference operator*() const { return *ptr; }
  pointer operator->() const { return *ptr; }

  self& operator++() { ++ptr; return *this; }
  self operator++(int) { self tmp = *this; ++ptr; return *this; }

  self& operator--() { --ptr; return *this; }
  self operator--(int) { self tmp = *this; --ptr; return *this; }

  bool operator==(const self& rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const self& rhs) const { return ptr != rhs.ptr; }

};

template <typename T>
class MyIter : public iterator_base<T, T&, T*> {
 public: 
  using self = MyIter;
  using Base = iterator_base<T, T&, T*>;

  MyIter(typename Base::pointer p = nullptr) : Base(p) {}
  MyIter(const Base::iterator& x) : Base(x) {}

  self& operator++() {
    ++Base::ptr;
    return *this;
  }
  self operator++(int) {
    self tmp = *this;
    ++Base::ptr;
    return *this;
  }
};

template <typename T>
class MyConstIter : public iterator_base<T, const T&, const T*> {
 public: 
  using self = MyConstIter;
  using Base = iterator_base<T, const T&, const T*>;

  MyConstIter(typename Base::pointer p = nullptr) : Base(p) {}
  MyConstIter(const Base::iterator& x) : Base(x) {}

  const self& operator++() {
    ++Base::ptr;
    return *this;
  }
  self operator++(int) {
    self tmp = *this;
    ++Base::ptr;
    return *this;
  }
};

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
  int arr[] = { 4, 2, 1, 6, 0 };
  MyIter<int> begin(arr);
  MyIter<int> end(arr + 5);
  MyConstIter<int> _begin(arr);
  MyConstIter<int> _end(arr + 5);

  for (auto it = _begin; it != _end; ++it) {
    cout << *it << " "; 
  }
  cout << endl;

  // 只能使用 mutable iterator 进行修改
  for (auto it = begin; it != end; ++it) {
    *it += 1;
  }

  for (auto it = _begin; it != _end; ++it) {
    cout << *it << " "; 
  }
  cout << endl;

  // 在输出结果中 const int 和 int 都表示为 i
  type_check(begin);
  type_check(_begin);

  MyIter<int> it(arr + 2);
  MyConstIter<int> it_const(it);
  cout << *it_const << endl;

  system("pause");
  return 0;
}