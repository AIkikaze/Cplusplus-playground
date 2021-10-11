/*
 * author: wenwenziy
 * title: 链表奇偶节点重构23
 * last edited: 2021-10-11
 * function: rebuild
 * trick: 快慢指针变形,注意链表节点的链接次序
 */
#include <iostream>
#include <cstdio>
using namespace std;
typedef struct list_node {
	int key;
	list_node *nex;
	list_node (int _k = 0):key(_k), nex(NULL) {  } 
	~list_node () { nex = NULL; }
}Node;

void rebuild (Node *hd) {
	Node *slow = hd, *fast = hd->nex;
	while (fast != NULL && fast->nex != NULL) {
		Node *tmp = fast->nex->nex;
		fast->nex->nex = slow->nex; // 再保证fast->n链接slow->n
		slow->nex = fast->nex;      // 最后slow连上fast->nex
		fast->nex = tmp;            // fast连上下一个偶数节点
		fast = tmp;                 // fast 前进到下一个偶数节点
		slow = slow->nex;           // slow 前进到下一个节点
	}
}

int main () {
	Node *head;
	int n, *arr;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++) 
		scanf("%d", &arr[i]);
	head = new Node(arr[0]);
	Node *cur = head;
	for (int i = 1; i < n; i++) {
		Node *u = new Node(arr[i]);
		cur->nex = u;
		cur = u;
	}
	rebuild (head);
	for (Node *p = head; p != NULL; p = p->nex) 
		printf("%d ", p->key);
	return 0;
}
