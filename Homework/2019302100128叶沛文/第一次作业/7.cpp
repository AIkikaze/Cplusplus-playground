/*
 * author: wenwenziy
 * title: 对向量实现冒泡和梳排序算法7
 * last edited: 2021-10-09
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
    void update_cap ();
    // feature 2: 调试函数
    void dbg ();
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
		// 向量插入排序
		void insert_sort (const int &);
		// 向量冒泡排序
		void bubble_sort (const int &);
		// 向量梳排序
		void ex_bubble_sort (const int &);
};

template<typename T>
void myVector<T>::bubble_sort (const int &len) {
    for (int i = 0; i < len-1; i++) 
      for (int j = 0; j < len-i-1; j++) 
        if (base[j] > base[j+1])
          swap(base[j], base[j+1]);
}

template<typename T>
void myVector<T>::ex_bubble_sort (const int &len) {
	int gap = len-1;
	while (1) {
		for (int j = 0; j < len-gap; j++) 
			if (base[j] > base[j+gap])
				swap(base[j], base[j+gap]);
		gap = (int)(gap/1.3);
		if (!gap) break;
	}
}

int main () {
	int n, *arr;
	myVector<int> V;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++) 
		scanf("%d", &arr[i]);
	V.init(arr, n);
	printf("排序前: "), V.disp();
	V.ex_bubble_sort(n);
	printf("排序后: "), V.disp();
	delete [] arr;
	return 0;
}
