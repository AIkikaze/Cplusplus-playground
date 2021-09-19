#include <cstring>
#include <iostream>
#include <cstdio>
#include <pthread.h>
using namespace std;
const int list_max_len = 1e6;
const int char_max_len = 1e2;

struct node {
	char *name, *number;
	node *nex;

	node () {
		number = new char [char_max_len];
		name = new char [char_max_len];
		nex = NULL;
	}

	void operator = (const node &u) {
		memcpy(number, u.number, strlen(u.number)+1);
		memcpy(name, u.name, strlen(u.name)+1);
	}

	~node () {
		delete [] number;
		delete [] name;
		nex = NULL;
	}
};

class list {
	public:
		node *head;
		int capacity;

		list () {
			head = NULL;
			capacity = list_max_len;
		}

		~list () {
			node *p, *pre;
			pre = head;
			p = head->nex;
			while (p != NULL) {
				delete pre;
				pre = p;
				p = p->nex;
			}
			delete pre;
		}
		// 返回一个 bool 变量，若list为空则返回true，否则返回false
		bool isEmpty ();
		// 返回一个整形变量，其值为list的长度
		int listLength ();
		// 从头输出整个列表
		void dispList (int hl);
		// 输出列表的第i个元素，返回一个bool变量，若成功输出则返回true否则为false
		bool getItem (const int &i, node &e);
		// 按number从头查找第一个与e相等的结点，若存在这样的点，则返回true，否则返回false
		int checkItem (const char *str);
		// 在列表第i-1个元素之后插入数据元素，返回是否成功插入
		bool insertList (const int &i, const node &inode);
		// 删除列表的第i个元素，并返回是否成功删除
		bool deleteList (const int &i);
};

bool list::isEmpty() {
	return head == NULL ? true : false;
}

int list::listLength() {
	int length = 0;
	node *p = head;
	while (p != NULL) {
		length++;
		p = p->nex;
	}
	return length;
}

void list::dispList(int hl = 0) {
	node *p = head;
	int t = 0;
	while (p != NULL) {
		t++;
		if (t == hl)
			printf("<%d>: %s %s\n", t, p->number, p->name);
		else 
			printf("[%d]: %s %s\n", t, p->number, p->name);
		p = p->nex;
	}
}

bool list::getItem(const int &i, node &e) {
	node *p;
	p = head;
	if (i < 1 || p == NULL) return false;
	for (int t = 1; t < i; t++) {
		if (p->nex != NULL)
			p = p->nex;
		else 
			return false;
	}
	e = *p;
	return true;
}

int list::checkItem(const char *str) {
	node *p = head;
	int t = 0;
	while (p != NULL) {
		t++;
		if (strcmp(p->number, str) == 0)
			return t;
		else 
			p = p->nex;
	}
	return 0;
}

bool list::insertList(const int &i, const node &inode) {
	node *p, *newp = new node();
	p = head;
	*newp = inode;
	if (i < 1) {
		delete newp;
		return false;
	} 
	if (i == 1 || head == NULL) {		
		newp->nex = head;
		head = newp;
		return true;
	}
	for (int t = 1; t < i-1; t++) {
		if (p != NULL)
			p = p->nex;
		else 
			return false;
	}
	newp->nex = p->nex;
	p->nex = newp;
	this->dispList();
	return true;
}

bool list::deleteList(const int &i) {
	node *p, *pre;
	pre = NULL;
	p = head;

	if (i < 1 || head == NULL)
		return false;
	if (i == 1) {
		head = p->nex;
		delete p;
		return true;
	}
	for (int t = 1; t < i; t++) {
		if (p != NULL)
			pre = p, p = p->nex;
		else 
			return false;
	}
	pre->nex = p->nex;
	delete p;
	return true;
}

int main () {
	list L;
	int N;
	scanf("%d", &N);
	for (int i = 1; i <= N; i++) {
		node v;
		scanf("%s%s", v.number, v.name);
		L.insertList(i, v);
	}
	L.dispList();
	int n;
	while (scanf("%d", &n) && n) {
		node v;
		int i = 0;
		bool flag = 0;
		switch (n) {
			case 1: 
				printf("insert item:");
				scanf("%d%s%s", &i, v.number, v.name);
				flag = L.insertList(i, v);
				break;
			case 2:
				printf("display list:\n");
				L.dispList();
				flag = 1;
				break;
			case 3:
				printf("delete item:");
				scanf("%d", &i);
				flag = L.deleteList(i);
				break;
			case 4:
				printf("get the item:\n");
				scanf("%d", &i);
				flag = L.getItem(i, v);
				printf("%s %s\n", v.number, v.name);
				break;
			case 5:
				printf("check the item:");
				scanf("%s", v.number);
				if (L.checkItem(v.number)) {
					L.dispList(L.checkItem(v.number));
				}
				else 
					printf("unfound\n");
				flag = 1;
				break;
		}
		if (!flag)
			printf("unkown order!");
	}
	return 0;
}
