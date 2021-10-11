/*
 * author: wenwenziy
 * title: 数组去重(大于2次出现)
 * last edited: 2021-10-11
 * run case: 见注释
 * function: unique_arr, sort
 * method: 快慢指针法,便于本地修改数组节省空间
 * tip: 先排序再去重,否则目前看来并没效率很高
 * 去重算法,我能想到的就是 筛法 和 桶排序,前者
 * 耗时,后者浪费空间
 */ 
#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;

// 10 2 7 7 7 8 8 1 1 1 1
//*1 *1  1  1  2  7  7  7  8  8  slow:0 fast:1 count:1
// 1 *1 *1  1  2  7  7  7  8  8  slow:1 fast:2 count:2
// 1 *1  1 *1  2  7  7  7  8  8  slow:1 fast:3 count:2
// 1 *1  1  1 *2  7  7  7  8  8  slow:1 fast:4 count:2
// 1  1 *2  1  2 *7  7  7  8  8  slow:2 fast:5 count:1
// 1  1  2 *7  2  7 *7  7  8  8  slow:3 fast:6 count:1
// 1  1  2  7 *7  7  7 *7  8  8  slow:4 fast:7 count:2
// 1  1  2  7 *7  7  7  7 *8  8  slow:4 fast:8 count:2
// 1  1  2  7  7 *8  7  7  8 *8  slow:5 fast:9 count:1
// 1  1  2  7  7  8 *8  7  8  8  slow:6 fast:10 count:2
// 1 1 1 1 2 7 7 7 8 8
// 1 1 2 7 7 8 8 %

int unique_arr (int *base, int len) {
	int slow, fast, count;
	slow = 0;
	count = fast = 1;
	while (fast < len) {
		if (base[slow] == base[fast] && count < 2) {
			count++;
			base[++slow] = base[fast];
		}
		else if (base[slow] != base[fast]) {
			count = 1;
			base[++slow] = base[fast];
		}
		fast++;
	}
	return slow+1;
}

int main () {
	int n, *arr;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++) 
		scanf("%d", &arr[i]);
	sort(arr, arr+n);
	int _n = unique_arr(arr, n);
	for (int i = 0; i < _n; i++) 
		printf("%d ", arr[i]);
	return 0;
}
