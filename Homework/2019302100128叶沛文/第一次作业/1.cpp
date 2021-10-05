/* 
 * author: wenwenziy
 * title: 动态向量实现1-1
 * class: myVector
 * last edited: 2021-10-5
 * functions and features: 见注释
 * warning: 虽然使用了模板类，但部分函数是默认按int来处理的
 * （如输入和输出部分），故还不能其他类型作为vector的参数
 */
#include <iostream>
#include <cstdio>
using namespace std;
const unsigned int maxCapacityofvector = 1e7;

template<typename T>
class myVector {
  private:
    int capacity, len;
    T *base;

  public:
    myVector () {
      capacity = 0;
      len = 0;
      base = NULL;
    }
    ~myVector () {
      if (!empty()) clear();
      else delete [] base;
    }
    // feature 1: 实现vector容量的动态缩减
    void update_cap () {
      capacity = capacity<<1;
      if (capacity > maxCapacityofvector) 
        throw "myVector::capacity of vector is too large";
      // 注意这里和init初始化的vector有所不同，没有赋予初始值
      // 完整的写法（特别是在T为更为复杂的类时）应该声明一个新的myVector
      // 并重载拷贝函数，再将新的myVector拷贝到this
      T *_base = new T[capacity]; 
      // 注意字符串的结尾会有一个结束字符，所以才需要增加一位
      // 这里不需要
      memcpy(_base, base, len*sizeof(T));
      base = _base;
    }
    // feature 2: 调试函数
    void dbg () {
      cout << "当前vector容量:" << capacity << endl;
      cout << "当前vector长度:" << len << endl;
      cout << "当前vector头指针:" << &base << endl;
      cout << "当前vector的内容:";
      try {
        disp();
      } catch(const char *str) {
        printf("%s\n", str);
      }
    }
    // feature 3: 控制命令
    void setup () {
      char ch;
      int n, pos, key, cap;
      int *arr = NULL;
      while (ch != 'q') {
        printf("(myVector)");
        cin >> ch;
        switch (ch) {
          case 'h':
            printf("命令列表：\n");
            printf("0: 创建一个容量为cap，初值为0的空vector\n");
            printf("1: 用一个数组来初始化 vector\n");
            printf("2: 按顺序输出整个 vector 所有的元素\n");
            printf("3: 插入元素 value 到指定位置 pos\n");
            printf("4: 返回 vector 中第 pos 个元素的值\n");
            printf("5: 删除指定位置 pos 元素\n");
            printf("h: 查询命令列表\n");
            printf("d: debug\n");
            printf("c: 清空 vector\n");
            printf("q: 退出\n");
            break;
          case '0':
            scanf("%d%d", &cap, &key);
            init(cap, key);
            break;
          case '1':
            scanf("%d", &n);
            arr = new int[n];
            for (int i = 0; i < n; i++) 
              scanf("%d", &arr[i]);
            init(arr, n);
            break;
          case '2':
            try{
              disp();
            } catch(const char *str) {
              printf("%s\n", str);
            }
            break;
          case '3':
            scanf("%d%d", &pos, &key);
            try {
              insert_item(pos, key);
            } catch (const char *str) {
              printf("%s\n", str);
            }
            break;
          case '4':
            scanf("%d", &pos);
            try {
              printf("%d\n", check_item(pos));
            } catch (const char *str) {
              printf("%s\n", str);
            }
            break;
          case '5':
            scanf("%d", &pos);
            try {
              delete_item(pos);
            } catch (const char *str) {
              printf("%s\n", str);
            }
            break;
          case 'd':
            dbg();
            break;
          case 'c':
            clear();
            break;
          case 'q':
            break;
        }
      }
      if (arr != NULL) delete []arr;
    }
    // feature 4:用一个数组来初始化 vector
    void init (const int arr[], int _len) {
      init(_len);
      len = _len;
      for (int i = 0; i < _len; i++) 
        base[i] = arr[i];
    } 
    //a. vector 初始化并创建一个空表，容量为cap
    void init (const int &cap, int key = 0) {
      capacity = cap;
      base = new T[cap];
      for (int i = 0; i < cap; i++) 
        base[i] = key;
    }
    //b. 清除 vector 中的所有元素，释放存储空间，使之成为一个空表
    void clear () {
      capacity = 0;
      len = 0;
      if (base != NULL) delete [] base;
      base = NULL;
    }
    //c. 返回 vector 当前的长度，若为空则返回０
    int length () {
      return len;
    }
    //d. 判断 vector 是否为空，若为空则返回 1, 否则返回 0
    bool empty () {
      return len ? 0 : 1;
    }
    //e. 返回 vector 中第 pos 个元素的值
    T check_item (const int &pos) {
      if (!empty() && len >= pos)
        return base[pos-1];
      throw "myVector::pos out of range";
    }
    //f. 插入元素 value 到指定位置 pos
    void insert_item (const int &pos, const int &key) {
      if (len+1 >= pos) {
        if (len+1 >= capacity) update_cap();
        for (int i = len; i >= pos; i--)
          base[i] = base[i-1];
        base[pos-1] = key;
        len++;
      }
      else throw "myVector::pos out of range";
    }
    //g. 删除指定位置 pos 元素
    void delete_item (const int &pos) {
      if (pos && len >= pos) {
        for (int i = pos-1; i < len-1; i++)
          base[i] = base[i+1];
        len--;
      }
      else throw "myVector::pos out of range";
    }
    //h. 按顺序输出整个 vector 所有的元素
    void disp () {
      if (!empty()) {
        for (int i = 0; i < len; i++) 
          cout << base[i] << ' ';
        cout << endl;
      }
      else throw "myVector::vector is empty";
    }
};

int main () {
  myVector<int> v;
  v.setup();
  v.clear();
  return 0;
}
