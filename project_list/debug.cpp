#include <iostream>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cstdlib>
using namespace std;
const int listMaxCapacity = 1e6;
const int charMax = 1e2;
int tdele = 0;
int tnew = 0;

class listNode {
	public:
		int number;
		char *name;
		listNode *nex;

		listNode () {
			tnew++;
			number = 0;
			name = new char [10];
			nex = NULL;
		}

		listNode (const int &t_nu, const char *t_name) {
			name = new char [10];
			number = t_nu;
			memcpy(name, t_name, strlen(t_name)+1);
			nex = NULL;
		}

		~listNode () {
			listNode *r;
			r = this;
			printf("[%d]:\n", ++tdele);
			printf("before delete\n");
			while (r != NULL) {
				printf("%d %p %s\n", r->number, r, r->name);
				r = r->nex;
			}
			delete [] name;
			nex = NULL;
			r = this;
			printf("after delete\n");
			while (r != NULL) {
				printf("%d %p %s\n", r->number, r, r->name);
				r = r->nex;
			}
		}

		void operator=(const listNode &t) {
			number = t.number;
			memcpy(name, t.name, strlen(t.name)+1);
			nex = t.nex;
		}
};

class lisTol {
	public:
		int capacity;
		listNode *head;
		// 初始化线性表
		lisTol () {
			capacity = listMaxCapacity;
			head = new listNode();
		}
		// 销毁线性表
		~lisTol () {
			listNode *r = head->nex;
			listNode *pre = head;
			while (r != NULL) {
				delete pre;
				pre = r;
				r = r->nex;
			}
			delete pre;
		}
};

void dispList (lisTol &L) {
	listNode *r = L.head;
	int t = 0;
	while (r->number != 0) {
		printf("[%d]: %d %p %s\n", t+1, r->number, r, r->name);
		r = r->nex;
		t++;
	}
	printf("list end, the length of list = %d\n", t);
}

// 在第i个元素前插入新的节点
bool listInsert (lisTol &L, const int &i, const listNode &u) {
	listNode *r = L.head, *v = new listNode(), *pre = r;
	if (i < 0) return false;
	for (int t = 0; t < i; t++) {
		if (r != NULL)
			pre = r, r = r->nex;
		else 
			return false;
	}
	*v = u;
	v->nex = r;
	if (pre != r)
		pre->nex = v;
	else 
		L.head = v;
	return true;
}

bool listDelete (lisTol &L, const int &i) {
	listNode *r = L.head, *pre = r;
	for (int t = 1; t < i; t++) {
		if (r->nex->nex != NULL) 
			pre = r, r = r->nex;
		else 
			return false;
	}
	if (pre == r)
		L.head = r->nex;
	else
		pre->nex = r->nex;
	delete r;
	return true;
}

bool getNode (lisTol &L, const int &i, listNode &u) {
	listNode *r = L.head;
	for (int t = 1; t < i; t++) {
		if (r->nex->nex != NULL) 
			r = r->nex;
		else 
			return false;
	}
	u = *r;
	return true;
}

int getLength (lisTol &L) {
	listNode *r = L.head;
	int length = 0;
	while (r->number != 0) {
		r = r->nex;
		length++;
	}
	return length;
}

void sortList (lisTol &L) {
	listNode *r = L.head, *nl = r, *tmp;
	r = r->nex;
	while (r != NULL) {
		while (nl->nex != NULL && nl->number < r->number ) {
			nl = nl->nex;
		}
		tmp = r->nex;
		r->nex = nl->nex;
		nl->nex = r;
		r = tmp;
	}
}

int main () {
	lisTol L;
	int n, i, number;
	char *name = new char [10];
	listNode v;

	scanf("%d", &n);
	for (i = 0; i < n; i++) {
		scanf("%d%s", &number, name);
		listInsert(L, i, listNode(number, name));
	}
	dispList(L);
	while (scanf("%d", &n) && n) {
		switch (n) {
			case 1: 
				printf("insert item:");
				scanf("%d%d%s", &i, &number, name);
				listInsert(L, i, listNode(number, name));
				break;
			case 2:
				printf("display list:\n");
				dispList(L);
				break;
			case 3:
				printf("delete item:");
				scanf("%d", &i);
				listDelete(L, i);
				break;
			case 4:
				printf("check the item:\n");
				scanf("%d", &i);
				getNode(L, i, v);
				printf("%d %s", v.number, v.name);
				break;
			case 5:
				printf("sorting finished");
				sortList(L);
				break;
		}
	}
	printf("%d\n", tnew);

	delete [] name;
	return 0;
}
