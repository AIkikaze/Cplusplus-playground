/*
 * author: wenwenziy
 * title: 对向量实现实现二分查找9
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
			clear();
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
			if(key) for (int i = 0; i < cap; i++) 
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
		// 对向量实现实现二分查找
		// 返回大于等于key的元素位置的下界(数组指标从0开始)
		// 如果key大于所有元素,那么返回len
		int find (const T &key);
};

template<typename T>
void myVector<T>::insert_sort (const int &len) {
	T tmp; 
	int j;
	for (int i = 1; i < len; i++) {
		tmp = base[i];
		j = i;
		while (base[j-1] > tmp && j > 0) 
			base[j] = base[j-1], j--;
		base[j] = tmp;
	}
}

template<typename T>
int myVector<T>::find (const T &key) {
	int lb = 0, rb = len;
	while (lb < rb) {
		int mid = (lb+rb)>>1;
		if (base[mid] >= key) rb = mid;
		else lb = mid+1;
	}
	return lb;
}

int main () {
	int n, *arr, value;
	myVector<int> V;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++) 
		scanf("%d", &arr[i]);
	V.init(arr, n);
	V.insert_sort(n);
	V.disp();
	while (1) {
		scanf("%d", &value);
		printf("%d\n", V.find(value));
	}
	delete [] arr;
	return 0;
}
