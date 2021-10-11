/* 
 * author: wenwenziy
 * title: 循环队列实现5
 * last edited: 2021-10-08
 * class: 无封装,数组实现
 * functions and features: 见注释
 */
#include <iostream>
#include <cstdio>
#define fr(a,b,c) for(int i = a; i != b; c) 
using namespace std;
const int list_max_cap = 1e6;

int t[list_max_cap];
unsigned int front, rear, LEN;
// 判断队列是否为满
bool isfull () {
	return front == (rear+1)%LEN;
}
// 判断队列是否为空
bool isempty () {
	return front == rear;
}
// 入队
void push (const int &e) {
	if (!isfull()) t[rear++] = e, rear %= LEN; 
}
// 出队
void pop () {
	if (!isempty()) front++, front %= LEN;
}
// 输出整个队列
void disp () {
	fr(front, rear, i=(i+1)%LEN) printf("%d ", t[i]);
	printf("\n");
}

int main () {
	// 初始化链表
	int n, *arr;
	scanf("%d%d", &n, &LEN); arr = new int[n]; LEN++;
	fr(0, n, i++) {
		scanf("%d", &arr[i]);
		push(arr[i]);
	}
	while (1) {
		scanf("%d", &n);
		switch (n) {
			case 0: printf("0: 命令列表\n1: 入队\n2: 出队\n3: 输出整个队列\n");
							break;
			case 1: cin>>arr[0]; push(arr[0]); break;
			case 2: pop(); break;
			case 3: disp(); break;
		}
		if (n == -1) break;
	}
  return 0;
}

