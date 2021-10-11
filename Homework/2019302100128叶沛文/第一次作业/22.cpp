/*
 * author: wenwenziy
 * title: two sum 寻找和为sum的两数
 * last edited: 2021-10-11
 * tricks: 有序数组,两端查找;无序数组,哈希建表
 * tips: 为什么两端查找能够找到恰好和为sum的两
 * 数？- 可以证明：在有序数组中,两端查找的指针
 * 始终在和为sum两数的两边,不可能出现将结果漏
 * 过的情形.记结果指针为_i,_j,查找指针为i,j
 * 初始状态:
 * 0 1 2 3 ... n-1
 * i..._i..._j...j
 * 若在查找过程中,发生如下情形：
 * 1 0..._i.._j.i..j...n-1
 * 在此前必有 
 * 1 0..i._i.._j...j...n-1
 * 这时无论 ai + aj 与 taeget 的关系如何,i,j都不
 * 可能越过_i,_j
 *
 * 可以发现在i,j进行扫描移动的时候,始终保证target
 * 在待扫描的范围内,在此可以简单列举一下
 * 0...i...j...n-1
 * 此时,待扫描的ai+aj 有如下可能:
 * {a(i+1) + aj, a(i+2)+aj,..., a(j-1)+aj} 其最小值为 a(i+1)+aj
 * {ai + a(j-1), ai+a(j-2),..., ai+a(i+1)} 其最大值为 ai+a(j-1)
 */
#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;

int* find_sum (int *_begin, int *_end, int target) {
	int *answer = new int[4];
	int *lb, *rb;
	lb = _begin;
	rb = _end-1;
	while (lb < rb) {
		if (*lb + *rb == target) {
			answer[0] = *lb;
			answer[1] = lb-_begin;
			answer[2] = *rb;
			answer[3] = rb-_begin;
			return answer;
		}
		if (*lb + *rb > target) rb--;
		if (*lb + *rb < target) lb++;
	}
	delete [] answer;
	return NULL;
}

int main () {
	int n, *arr, target;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++) 
		scanf("%d", &arr[i]);
	sort(arr, arr+n);
	scanf("%d", &target);
	int *answer = find_sum(arr, arr+n, target);
	if (answer != NULL) {
		printf("找到两数为：%d %d\n", answer[0], answer[2]);
		printf("两数位置为：%d %d\n", answer[1], answer[3]);
	}
	else printf("Nr not found\n");
	return 0;
}
