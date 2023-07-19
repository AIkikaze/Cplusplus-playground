#include <iostream>
#include <vector>
using namespace std;

template <class Iterator>
struct my_iterator_traits {
  typedef typename Iterator::value_type value_type;
  typedef typename Iterator::reference reference;
  typedef typename Iterator::pointer pointer;
  typedef typename Iterator::difference_type difference_type;
  typedef typename Iterator::iterator_category iterator_category;
};

#define __val_(T) typename my_iterator_traits<T>::value_type
#define __ref_(T) typename my_iterator_traits<T>::reference
#define __ptr_(T) typename my_iterator_traits<T>::pointer
#define __diff_(T) typename my_iterator_traits<T>::difference_type
#define __iter_cate_(T) typename my_iterator_traits<T>::iterator_category

template <typename Iterator>
concept BidirectionalIterator = requires(Iterator it) {
  requires std::bidirectional_iterator<Iterator>;
  { it++ } -> std::same_as<Iterator>;
  { it-- } -> std::same_as<Iterator>;
};

template <BidirectionalIterator Iter>
class my_reverse_iterator {
 public:
  using self = my_reverse_iterator;
  using iteraTor_category = __iter_cate_(Iter);
  using value_type = __val_(Iter);
  using reference = __ref_(Iter);
  using pointer = __ptr_(Iter);
  using difference_type = __diff_(Iter);

  explicit my_reverse_iterator(Iter i) : iter(i) {}

  reference operator*() const {
    auto tmp = iter;
    return *(--tmp);
  }

  self& operator++() {
    --iter;
    return *this;
  }
  self operator++(int) {
    self tmp = *this;
    --iter;
    return tmp;
  }
  self& operator--() {
    ++iter;
    return *this;
  }
  self operator--(int) {
    self tmp = *this;
    ++iter;
    return tmp;
  }

  bool operator==(const self& rhs) const { return iter == rhs.iter; }
  bool operator!=(const self& rhs) const { return iter != rhs.iter; }

 private:
  Iter iter;
};

// 辅助函数，方便创建反向迭代器适配器
template <BidirectionalIterator Iter>
my_reverse_iterator<Iter> make_reverse_iterator(Iter it) {
  return my_reverse_iterator<Iter>(it);
}

int main() {
  vector<int> vec = {1, 2, 3, 4, 5};

  // 使用反向迭代器适配器反向遍历 vector
  for (auto it = make_reverse_iterator(vec.end());
       it != make_reverse_iterator(vec.begin()); ++it) {
    cout << *it << " ";
  }
  cout << endl;

  system("pause");

  return 0;
}