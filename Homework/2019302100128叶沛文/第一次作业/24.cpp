/*
 * author: wenweniy
 * title: 计算更大值的位置24
 * last edited: 2021-10-11
 * trick: 简简单单栈模拟(维护一个数值单增的栈)
 * tips: 注意栈不仅要存数值,更要存上位置
 */
#include <iostream>
#include <cstdio>
#include <ctime>
using namespace std;
const int arrMax = 1e2+1; 

int main () {
  srand((unsigned)time(NULL));
  int n, *arr;
  scanf("%d", &n);
  arr = new int[n];
  for (int i = 0; i < n; i++) 
		//scanf("%d", &arr[i]);
    arr[i] = rand()%arrMax;
	int *st = new int[n<<1], *upfound = new int[n];
	int top = -1;
	for (int i = 0; i < n; i++) upfound[i] = 0;
	for (int i = 0; i < n; i++) {
		/* getchar(); */
		/* for (int i = 0; i <= top; i++) */
		/* 	printf("%d-%d ", st[i], st[i+n]); */
		/* printf("\n"); */
		if (top == -1 || arr[i] <= st[top]) 
			st[++top] = arr[i], st[top+n] = i;
		else {
			while (top > -1 && arr[i] > st[top]) {
				upfound[st[top+n]] = i - st[top+n];
				top--;
			}
			st[++top] = arr[i];
			st[top+n] = i;
		}
	}
	for (int i = 0; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");
	for (int i = 0; i < n; i++)
		printf("%d ", upfound[i]);
	return 0;
}
