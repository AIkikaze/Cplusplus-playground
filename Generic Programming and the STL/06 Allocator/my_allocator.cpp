#include <iostream>
#include <new>
#include <vector>
#include <memory>
using namespace std;

namespace wenzy {

template <class T>
inline T*  _allocate(ptrdiff_t size, T*) {
  set_new_handler(0);
  T* target = (T*)(::operator new((size_t)(size * sizeof (T))));
  if (target == 0) {
    cerr << "out of memory" << endl;
    exit(0);
  }
  return target;
}

template <class T>
inline void _deallocate(T* buffer) {
  ::operator delete(buffer);
}

template <class T1, class T2>
inline void _construct(T1* p, const T2& value) {
  new(p) T1(value);
}

template <class T>
inline void _destroy(T* ptr) {
  ptr->~T();
}

template <class T>
class allocator {
 public:
  typedef T          value_type;
  typedef T*         pointer;
  typedef const T*   const_pointer;
  typedef T&         reference;
  typedef const T&   const_reference;
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;

  template <class U>
  struct rebind {
    typedef allocator<U> other;
  };

  pointer allocate(size_type n, const void* hit = 0) {
    return _allocate((difference_type)n, (pointer)0);
  }

  void deallocate(pointer p, size_type n) { _deallocate(p); }

  void construct(pointer p, const T& value) { _construct(p, value); }

  void destory(pointer p) { _destroy(p); }

  pointer address(reference x) { return (pointer)&x; }

  const_pointer const_address(const_reference x) { return (const_pointer)&x; }

  size_type max_size() const { return size_type(UINT_MAX/sizeof (T)); }
};
}

int g1;
int g2 = 0;
int g3 = 1;
const int g4 = 1;
constexpr int g5 = 1;
static int g6 = 1;

int main() {
  vector<int, wenzy::allocator<int>> v = { 0, 1, 2, 3, 4, 5, 6, 7 };
  for (auto &val : v) {
    cout << val << " ";
  }
  cout << endl;

  static int s1 = 1;
  static int s2 = 1;
  int* p1 = new int;
  int* p2 = new int;
  int l0;
  int l1 = 1;
  int l2 = 1;

  // 代码区 .text
  int (*mainPtr) () = &main;
  cout << "代码区 main = " << mainPtr << endl;
  // 全局变量 
  cout << "int &g1 = " << &g1 << endl;
  cout << "int &g2 = " << &g2 << endl;
  cout << "int &g3 = " << &g3 << endl;
  cout << "int &g4 = " << &g4 << endl;
  cout << "int &g5 = " << &g5 << endl;
  cout << "int &g6 = " << &g6 << endl;
  // 局部静态
  cout << "int &s1 = " << &s1 << endl;
  cout << "int &s2 = " << &s2 << endl;  // 堆内存
  cout << "p1 = " << p1 << endl;
  cout << "p2 = " << p2 << endl;
  cout << "p1 - p2 = " << p1 - p2 << endl;
  // 局部栈
  cout << "int &l0 = " << &l0 << endl;
  cout << "int &l1 = " << &l1 << endl;
  cout << "int &l2 = " << &l2 << endl;

  system("pause");

  return 0;
}