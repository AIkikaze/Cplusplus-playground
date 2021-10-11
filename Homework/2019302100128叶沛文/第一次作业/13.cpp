/*
 * author: wenwenziy
 * title: 实现快速排序的函数，对向量进行排序
 * last edited: 2021-10-10
 */
#include <iostream>
#include <cstdio>
using namespace std;

template<typename T>
struct myVector {
	int capacity, len;
	T *base;
	myVector (int cap = 1e6) {
		capacity = cap;
		len = 0;
		base = new T[cap];
	}
	T* begin () { return base; }
	T* end () { return base+len; }
};

// 快速排序 模板+左闭右开
template<typename T>
void quick_sort (T *front, T *end) {
	if (front+1 >= end) return;
	T key = *front;
	T *lidx = front;
	T *ridx = end-1;
	while (lidx < ridx) {
		while (ridx > lidx && *ridx >= key)
			ridx--;
		*lidx = *ridx;
		while (lidx < ridx && *lidx < key)
			lidx++;
		*ridx = *lidx;
	}
	*lidx = key;
	// 注意中间的key已经放好位置就可以不用在排了
	quick_sort(front, lidx);
	quick_sort(lidx+1, end);
}

int main () {
	myVector<int> V;
	int n, value;
	scanf("%d", &n);
	for (int i = 0; i < n; i++)
		scanf("%d", &value), V.base[V.len++] = value;
	quick_sort(V.begin(), V.end());
	for (int *p = V.begin(); p < V.end(); p++)
		printf("%d ", *p);
	return 0;
}
