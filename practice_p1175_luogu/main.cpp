#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;

template<typename T>
class myDeque {
#define M(A) (A+capacity)%capacity
  private:
    T *base;
    int front, rear;
    int capacity;
    unsigned int length;
  public:
    myDeque () {
      capacity = 1e2;
      base = new T[capacity];
      front = 1;
      rear = 0;
      length = 0;
    }
    ~myDeque () {
      delete [] base;
      front = 1;
      rear = 0;
      length = 0;
    }
    void resize (const int &_capacity) {
      T *_base = new T[_capacity];
      memcpy(_base, base, capacity);
      capacity = _capacity;
      delete [] base;
      base = _base;
    }
    bool empty () {
      return !length;
    }
    bool fullszie () {
      return M(rear + 1) == front && !length;
    }
    int getLength () {
      return length;
    }
    void push_front (const T &e) {
      if (fullszie())
        resize(capacity<<1);
      base[M(--front)] = e;
      length++;
    }
    void push_back (const T &e) {
      if (fullszie())
        resize(capacity<<1);
      base[M(++rear)] = e;
      length++;
    }
    bool pop_front () {
      if (!empty()) {
        length--;
        front++;
        return true;
      }
      else
        return false;
    }
    bool pop_back () {
      if (!empty())  {
        length--;
        rear--;
        return true;
      }
      else
        return false;
    }
    T get_front () {
      return base[M(front)];
    }
    T get_back () {
      return base[M(rear)];
    }
    void dispDeque () {
      int _f, _r, _length;
      _f = front;
      _r = rear;
      _length = length;
      while (_length) {
        char ch = base[M(_f)].ch;
        if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^')
          printf("%c ", ch);
        else
          printf("%ld ", base[M(_f)].n);
        _f++;
        _length--;
      }
      printf("\n");
    }
    void dbg ();
};

struct calcuelem {
  long n;
  char ch;
  calcuelem () {
    n = 0;
    ch = '\0';
  }
};

template<typename T>
void myDeque<T>::dbg () {
  int _length, _f;
  _f = front;
  _length = length;
  printf("[%d]: ", _length);
  while (_length) {
    cout << base[M(_f++)] << ' ';
    _length--;
  }
  printf("out\n");
}

void PostExp (char *exp, myDeque<calcuelem> *pexp) {
  myDeque<char> s;

  while (*exp) {
    calcuelem t;
    if (*exp >= '0' && *exp <= '9') {
      while (*exp >= '0' && *exp <= '9') {
        t.n *= 10;
        t.n += *exp++ - '0';
      }
      pexp->push_back(t);
    }
    if (*exp == '(') {
      s.push_front(*exp);
    }
    if (*exp == ')') {
      // 如果栈中有'('我们需要把'('从栈中弹出
      for (t.ch = s.get_front(); t.ch != '(';) {
        pexp->push_back(t);
        s.pop_front();
        t.ch = s.get_front();
      }
      s.pop_front();
    }
    if (*exp == '^') {
      s.push_front(*exp);
    }
    if (*exp == '+' || *exp == '-') {
      if (!s.empty()) {
        for (t.ch = s.get_front(); t.ch != '('; ) {
          pexp->push_back(t);
          s.pop_front();
          if (s.empty()) break;
          t.ch = s.get_front();
        }
      }
      s.push_front(*exp);
    }
    if (*exp == '*' || *exp == '/') {
      if (!s.empty()) {
        for (t.ch = s.get_front(); t.ch == '*' || t.ch == '/' || t.ch == '^';) {
          pexp->push_back(t);
          s.pop_front();
          if (s.empty()) break;
          t.ch = s.get_front();
        }
      }
      s.push_front(*exp);
    }
    exp++;
  }
  while (!s.empty()) {
    calcuelem t;
    t.ch = s.get_front();
    pexp->push_back(t);
    s.pop_front();
  }
}

void soluExp (myDeque<calcuelem> &T) {
  myDeque<int> s;
  int k1, k2;

  while (T.getLength() > 0) {
    calcuelem t;
    t = T.get_front();
    if (t.ch == '+' || t.ch == '-' || t.ch == '*' || t.ch == '/' || t.ch == '^') {
      T.pop_front();
      k1 = s.get_front();
      s.pop_front();
      k2 = s.get_front();
      s.pop_front();
      switch (t.ch) {
        case '+':
          t.n = k1+k2;
          break;
        case '-':
          t.n = k2-k1;
          break;
        case '*':
          t.n = k1*k2;
          break;
        case '/':
          t.n = k2/k1;
          break;
        case '^':
          t.n = pow(k2, k1);
          break;
      }
      t.ch = '\0';
      T.push_front(t);
      while (!s.empty()) {
        t.n = s.get_front();
        T.push_front(t);
        s.pop_front();
      }
      T.dispDeque();
    }
    else {
      s.push_front(t.n);
      T.pop_front();
    }
    // T.dispDeque();
  }
}

int main () {
  myDeque<calcuelem> q;
  char _exp[105], _pexp[105];
  memset(_exp, 0, 105);
  memset(_pexp, 0, 105);

  //printf("give me a infex expression without and <space>\n");
  scanf("%s", _exp);
  PostExp(_exp, &q);
  q.dispDeque();
  //printf("emmm\n");
  soluExp(q);
  return 0;
}
