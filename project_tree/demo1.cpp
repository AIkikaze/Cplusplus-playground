/*
 * author: wenwenziy
 * last edited: 2021-11-05
 * title: 洛谷练习 p1035 新二叉树 
 * features: 链表存储；栈实现遍历；二叉树
 */
#include <iostream>
#include <cstdio>
#include <stack>
using namespace std;

template <typename T> class Tree;
template <typename T>
class Treenode {
private:
	T key;
	Treenode<T> *lson, *rson, *fa;
	friend class Tree<T>;
public:
	Treenode (int _key = 0) {
		fa = lson = rson = NULL;
		key = T(_key);
	}
	~Treenode () {
		lson = rson = NULL;
		if (fa != NULL) {
			if (fa->lson == this) 
				fa->lson = NULL;
			else 
				fa->rson = NULL;
		}
		fa = NULL;
	}
};

template<typename T>
class Tree {
#define Node Treenode<T>
public:
	Node *rt;

	Tree () {
		rt = NULL;
	}
	~Tree () {
		clear(rt);
		rt = NULL;
	}

	void insert (T key, T l, T r) {
		if (rt == NULL) {
			rt = new Node(key);
			if (l != '*') rt->lson = new Node(l);
			if (r != '*') rt->rson = new Node(r);
			rt->fa = rt;
			return;
		}
		stack<Node *> st;
		st.push(rt);
		while (!st.empty()) {
			Node *cur = st.top();
			if (cur != NULL) {
				st.push(cur->lson);
			}
			else {
				while (!st.empty() && st.top() == NULL) st.pop();
				while (!st.empty() && st.top()->rson == NULL) {
					if (st.top()->key == key) {
						if (l != '*') st.top()->lson = new Node(l);
						if (r != '*') st.top()->rson = new Node(r);
					}
					st.pop();
				}
				if (st.empty()) break;
				cur = st.top();
				st.pop();
				st.push(cur->rson);
			}
		}
	}

	void predisp (Node *root) { 
		if (root == NULL) return;
		stack<Node *> st;
		st.push(root);
		while (!st.empty()) { // 按前序遍历输出root子树
			Node *cur = st.top();
			if (cur != NULL) {
				printf("%c", cur->key);
				st.pop();
				st.push(cur->rson);
				st.push(cur->lson);
			}
			else {
				while (!st.empty() && st.top() == NULL) st.pop();
			}
		}
	}

	void clear (Node *root) {
		if (root == NULL) return;
		stack<Node *> st;
		st.push(root);
		while (!st.empty()) {  // 按中序遍历删除以root为根的子树
			Node *cur = st.top();
			if (cur != NULL) {
				st.pop();
				st.push(cur->rson);
				st.push(cur->lson);
				delete cur;
			}
			else {
				while (!st.empty() && st.top() == NULL) st.pop();
			}
		}
	}
#undef Node
};

int main () {
	Tree<char> T;
	int n;
	char ch[4];
	scanf("%d", &n);
	for (int i = 0; i < n; i++) {
		scanf("%s", ch);
		T.insert(ch[0], ch[1], ch[2]);
	}
	T.predisp(T.rt);
	printf("\n");
	return 0;
}
