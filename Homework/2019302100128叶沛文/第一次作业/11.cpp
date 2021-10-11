/*
 * author: wenwenziy
 * title: 利用向量和队列解决 Joseph 问题
 * last edited: 2021-10-10
 * question: 见维基百科 Joseph 问题
 * initial value: 问题初值
 * 人数为 n, 跳跃人数为 m, 起点为 sta
 * 约定犯人编号为 1,2,3,...,n
 */
#include <iostream>
#include <cstdio>
using namespace std;

namespace vector_method {
const int vec_max = 1e6;
int n, m, sta;    // 问题初值
int nr[vec_max];  // 犯人编号
int alive, prev;  // 当前存活犯人个数,前一个被杀的人的位置

void main () {
	scanf("%d%d%d", &n, &m, &sta);
	for (int i = 0; i < n; i++) nr[i] = i+1;
	alive = n;
	prev = 0;
	while (alive > 1) {
		prev = (prev+m-1)%alive;
		printf("%d ", nr[prev]);
		for (int i = prev; i < alive-1; i++)
			nr[i] = nr[i+1];
		alive--;
	}
}
}

namespace list_method {
typedef struct list_node {
	int Nr;
	list_node *nex;
	list_node (int _Nr): Nr(_Nr), nex(NULL) {  }
	~list_node () { 
		nex = NULL;
	}
}Node;
Node *head, *cur, *prev, *start;
int n, m, sta; // 问题初值
int count;     // 计数项

void main () {
	scanf("%d%d%d", &n, &m, &sta);
	// 为了和计数项一一对应 head作空节点
	head = new Node(0);
	head->nex = head;
	cur = head;
	for (int i = n; i > 0; i--) {
		Node *u = new Node(i);
		u->nex = cur;
		cur = u;
		if (i == sta) start = u;
		if (i == 1) head->nex = u; // 构成循环链表
	}
	cur = prev = head;
	while (head->nex->nex != head) {
		if (count == m) {   // count 对应 cur
			// 计数数到 m 进行裁决, 删除节点cur
			printf("%d ", cur->Nr);
			prev->nex = cur->nex;
			delete cur;
			cur = prev->nex;
			if (cur != head) count = 1;  // 注意细节 
			else count = 0;              // head为空节点需跳过
		}
		else {
			if (cur == prev) cur = start;
			else prev = cur, cur = cur->nex;
			if (cur != head) count++;
		}
		// printf("%d: %d\n", count, cur->Nr);
	}
}
}

int main () {
	// vector_method::main();
	list_method::main();
	return 0;
}

