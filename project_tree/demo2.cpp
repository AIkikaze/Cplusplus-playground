/*
 * title: 二叉树的遍历和可视化
 * author: wenwenziy
 * last edited: 2021-11-06
 * features: 使用了大量的 static 关键字和递归特性
 * reference: 
 * [1] https://www.runoob.com/data-structures/binary-search-tree.html 
 * （菜鸟教程：二叉搜索树）
 * [2] https://www.zhihu.com/question/280630276
 * （知乎问题：如何画出二叉树的图形？)
 */
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <stack>
using namespace std;

template <typename Key, typename Val> class Tree;
#define TNode Treenode<_T, _S>
template <typename _T, typename _S>
class Treenode {
private:
	_T key;
	_S value;
	TNode *lson, *rson, *fa;
	friend class Tree<_T, _S>;

	int count (TNode *cur) {
		if (cur == NULL) return 0;
		return count(cur->lson) + count(cur->rson) + 1;
	}
public:
	Treenode (_T _key = 0, _S _value = 0) {
		fa = lson = rson = NULL;
		key = _key;
		value = _value;
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

	int count () {
		return count(this);
	}
};
#undef TNode

#define Node Treenode<Key, Val>
template <typename Key, typename Val>
class Tree {
private:
	Node *rt; // 二叉搜索树的根节点
	int count; // 树中的节点计数

	static Node* insert (Node *cur, Key key, Val value) {
		if (cur == NULL)
			return new Node(key, value);
		if (key == cur->key) 
			cur->value = value;
		else if (key < cur->key) 
			cur->lson = insert(cur->lson, key, value);
		else 
			cur->rson = insert(cur->rson, key, value);
		return cur;
	}

	static bool contain (Node *cur, Key key) {
		if (cur == NULL) 
			return false;
		if (key == cur->key) 
			return true;
		else if (key < cur->key)
			return contain(cur->lson, key);
		else // key > cur->key
			return contain(cur->rson, key);
	}

	static Val search (Node *cur, Key key) {
		if (cur == NULL) 
			return NULL;
		if (key == cur->key)
			return cur->value;
		else if (key < cur->key)
			return search(cur->lson, key);
		else // key > cur->key
			return search(cur->rson, key);
	}

	static void preOrder (Node *cur) {
		if (cur == NULL) return;
		cout << setw(4) << cur->key;
		preOrder(cur->lson);
		preOrder(cur->rson);
	}

	static void inOrder (Node *cur) {
		if (cur == NULL) return;
		inOrder(cur->lson);
		cout << setw(4) << cur->key;
		inOrder(cur->rson);
	}

	static void postOrder (Node *cur) {
		if (cur == NULL) return;
		postOrder(cur->lson);
		postOrder(cur->rson);
		cout << setw(4) << cur->key;
	}

	static void display (Node *cur) {
		bool *isRight = new bool[cur->count()];
		Display_main(cur, isRight, 0);
		delete [] isRight;
	}

	static void Display_main (Node *cur, bool isRight[], int indent) {
		if (indent > 0) {
			for (int i = 0; i < indent - 1; i++) {
				printf(isRight[i] ? "│   " : "    ");
			}
			printf(isRight[indent-1] ? "├── " : "└── ");
		}
		if (cur == NULL) {
			printf("(null)\n");
			return;
		}
		cout << setw(4) << cur->key << "->" << cur->value << endl;
		if (cur->lson == NULL && cur->rson == NULL)
			return;

		isRight[indent] = 1;
		Display_main(cur->lson, isRight, indent + 1);
		isRight[indent] = 0;
		Display_main(cur->rson, isRight, indent + 1);
	}

	void clear (Node *cur) {
		if (cur == NULL) return;
		if (cur->lson == NULL && cur->rson == NULL) {
			delete cur;
			return;
		}
		clear(cur->lson);
		clear(cur->rson);
	}
public:
	Tree () {
		rt = NULL;
		count = 0;
	}
	~Tree () {
		clear(rt);
		rt = NULL;
		count = 0;
	}

	int size () { 
		return count; 
	}

	bool isEmpty () { 
		return count == 0; 
	}

	void insert (Key key, Val value) { 
		count++;
		rt = insert(rt, key, value); 
	}
	bool contain (Key key) { 
		return contain(rt, key); 
	}

	Val search (Key key) { 
		return search(rt, key); 
	}

	void preOrder () { 
		preOrder(rt); 
		cout << endl;
	}
	void inOrder () { 
		inOrder(rt); 
		cout << endl;
	}

	void postOrder () { 
		postOrder(rt); 
		cout << endl;
	}

	void display () {
		display(rt);
	}
};
#undef Node

int main () {
	int n; 
	int number; 
	string name;
	cout << left; // 左对齐
	Tree<int, string> T; // 可使用其他类型的键值对
	scanf("%d", &n);
	for (int i = 0; i < n; i++)  {
		cin >> number >> name;
		T.insert(number, name);
	}
	T.preOrder();
	T.inOrder();
	T.postOrder();
	T.display();
	return 0;
}
