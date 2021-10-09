/*
 * author: wenwenziy
 * title: 对向量实现shell排序算法8
 * last edited: 2021-10-09
 * featrues: 排序对象为生成的一组随机数,可以选择不同的排序模式
 * mode 0: 将以0,1,2,...,n为下标的元素进行排序
 * mode 1: 将以1,2,4,...,2^i为下标的元素进行排序 (2^i < n)
 * mode 2: 将以0,3,7,...,2^i-1为下标的元素进行排序 (2^i-1 < n)
 * tips: 由于模式1,2下的数组过于稀疏,在选取 n > 1000 时效果会明
 * 显一些
 */
#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>
using namespace std;
const unsigned int maxCapacityofvector = 1e7;
const int arrMax = 100;

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
      clear();
    }
    // feature 1: 实现vector容量的动态缩减
    void update_cap ();
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
    // feature 3: 类的控制台
    void setup ();
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
      if (key) for (int i = 0; i < cap; i++) 
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
    int length ();
    //d. 判断 vector 是否为空，若为空则返回 1, 否则返回 0
    bool empty () { return !len; }
    //e. 返回 vector 中第 pos 个元素的值
    T check_item (const int &pos);
    //f. 插入元素 value 到指定位置 pos
    void insert_item (const int &pos, const int &key);
    //g. 删除指定位置 pos 元素
    void delete_item (const int &pos);
    //h. 按顺序输出整个 vector 所有的元素
    void disp () {
      if (!empty()) {
        for (int i = 0; i < len; i++) 
          cout << base[i] << ' ';
        cout << endl;
      }
      else throw "myVector::vector is empty";
    }
		void disp (const int *idx, int _len) {
			if (!empty()) {
				for (int i = 0; i < _len && idx[i] < len; i++)
					cout << base[idx[i]] << ' ';
				cout << endl;
			}
		}
		// 向量插入排序
		void insert_sort (const int &);
		// 向量冒泡排序
		void bubble_sort (const int &);
		// 向量梳排序
		void ex_bubble_sort (const int &);
		// 向量shell排序
		void shell_sort (const int *idx, int _len);
};

template<typename T>
void myVector<T>::shell_sort (const int *idx, int _len) {
	int h = 1, tmp, j;
	while (idx[_len-1] >= len && _len--);
	while (h < _len/3) h = h*3+1;
	while (h >= 1) {
		for (int i = h; i < _len; i++) {
			tmp = base[idx[i]];
			for (j = i; j>=h && base[idx[j-h]] > tmp; j-=h)
				base[idx[j]] = base[idx[j-h]];
			base[idx[j]] = tmp;
		}
		h = h/3;
	}
}

int main () {
  srand((unsigned)time(NULL));
	int n, mode, *arr, *idx_base[3];
	myVector<int> V;
	scanf("%d%d", &n, &mode);
	arr = new int[n];
	for (int i = 0; i < 3; i++)
		idx_base[i] = new int[n];
	for (int i = 0; i < n; i++)
		arr[i] = rand()%arrMax;
	for (int i = 0; i < n; i++) {
		idx_base[0][i] = i;
		idx_base[1][i] = (int)pow(2, i);
		idx_base[2][i] = (int)pow(2, i)-1;
	}
	V.init(arr, n);
	printf("排序前: "), V.disp(idx_base[mode], n);
	V.shell_sort(idx_base[mode], n);
	printf("排序后: "), V.disp(idx_base[mode], n);
	delete [] arr;
	return 0;
}
