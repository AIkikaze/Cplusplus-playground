/*
 * author: wenwenziy
 * title: 实现归并排序的函数，对向量进行排序
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

// 归并排序 模板+左闭右开
template<typename T>
void merge_sort (T *front, T *end) {
	if (front+1 == end) return;
	int i = 0;
	int len = end-front;
	T *mid = front+len/2;
	merge_sort(front, mid);
	merge_sort(mid, end);
	T *b = new T[len];
	T *lidx = front;
	T *ridx = mid;
	while (lidx < mid && ridx < end) {
		if (*lidx < *ridx) 
			b[i++] = *lidx++;
		else 
			b[i++] = *ridx++;
	}
	while (lidx < mid) b[i++] = *lidx++;
	while (ridx < end) b[i++] = *ridx++;
	for (i = 0; i < len; i++) *(front+i) = b[i];
	delete [] b;
}

int main () {
	myVector<int> V;
	int n, value;
	scanf("%d", &n);
	for (int i = 0; i < n; i++)
		scanf("%d", &value), V.base[V.len++] = value;
	merge_sort(V.begin(), V.end());
	for (int *p = V.begin(); p < V.end(); p++)
		printf("%d ", *p);
	return 0;
}
