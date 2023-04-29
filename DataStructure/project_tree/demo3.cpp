/*
 * author: wenwenziy
 * last edited: 2021-11-12
 * title: 类封装的平衡树实现
 */
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <stack>
using namespace std;

template <typename T> class Tree;
#define TNode Treenode<T>
template <typename T>
class Treenode {
	private:
		int _height;
		T key;
		TNode *lson, *rson, *fa;
		friend class Tree<T>;
		bool flag;

		int count (TNode *cur) {
			if (cur == NULL) return 0;
			return count(cur->lson) + count(cur->rson) + 1;
		}

	public:
		Treenode (T _key = 0) {
			lson = rson = NULL;
			fa = this;
			key = _key;
			_height = -1;
			flag = 0;
		}
		~Treenode () {
			lson = rson = fa = NULL;
		}

		int count () {
			return count(this);
		}
};

template <typename T>
class Tree {
	private:
		TNode *rt; // 二叉搜索树的根节点
		int count; // 树中的节点计数

		static int height (TNode *cur) {
			if (cur == NULL) return -1;
			if (cur->flag) return cur->_height;
			cur->flag = 1;
			return cur->_height = max(height(cur->lson), height(cur->rson)) + 1;
		}

		static int balance_factor (TNode *cur) {
			return height(cur->rson) - height(cur->lson);
		}

		static void sonex (TNode *cur, TNode *target) {
			if (cur->fa == cur) return;
			if (cur->fa->lson == cur)
				cur->fa->lson = target;
			else if (cur->fa->rson == cur)
				cur->fa->rson = target;
			target->fa = cur->fa;
		}

		static void lRotate (TNode *cur) {
			TNode *child = cur->rson;
			sonex(cur, child);
			cur->rson = child->lson;
			child->lson = cur;
			cur->fa = child;
			// 更新指标
			cur->flag = child->flag = 0;
		}

		static void rRotate (TNode *cur) {
			TNode *child = cur->lson;
			sonex(cur, child);
			cur->lson = child->rson;
			child->rson = cur;
			cur->fa = child;
			// 更新指标
			cur->flag = child->flag = 0;
		}

		static TNode* adjust (TNode *cur) {
			cur->flag = 0;
			if (abs(balance_factor(cur)) <= 1) return cur;
			if (balance_factor(cur) > 1) {
				if (balance_factor(cur->rson) < 0)
					rRotate(cur->rson);
				lRotate(cur);
			}
			else if (balance_factor(cur) < -1) {
				if (balance_factor(cur->lson) > 0)
					lRotate(cur->lson);
				rRotate(cur);
			}
			return cur->fa;
		}

		static TNode* lower_bound (TNode *cur) {
			TNode *tmp = cur->rson;
			if (tmp == NULL) return cur;
			while (tmp->lson != NULL)
				tmp = tmp->lson;
			return tmp;
		}

		static TNode* Insert (TNode *cur, T key) {
			if (cur == NULL)
				return new TNode(key);
			else if (key < cur->key)
				cur->lson = Insert(cur->lson, key), cur->lson->fa = cur;
			else
				cur->rson = Insert(cur->rson, key), cur->rson->fa = cur;
			return cur = adjust(cur);
		}

		static TNode* Delete (TNode *cur, T key) {
			if (cur == NULL) return NULL;
			if (key == cur->key) {
				if (cur->rson != NULL) {
					cur->key = lower_bound(cur)->key;
					cur->rson = Delete(cur->rson, cur->key);
					if (cur->rson != NULL) cur->rson->fa = cur;
					return cur = adjust(cur);
				}
				else {
					TNode *tmp = cur->lson;
					if (cur->lson != NULL) sonex(cur, cur->lson);
					delete cur;
					return tmp;
				}
			}
			else if (key < cur->key) {
				cur->lson = Delete(cur->lson, key);
				if (cur->lson != NULL) cur->lson->fa = cur;
			}
			else {
				cur->rson = Delete(cur->rson, key);
				if (cur->rson != NULL) cur->rson->fa = cur;
			}
			return cur = adjust(cur);
		}

		static bool contain (TNode *cur, T key) {
			if (cur == NULL)
				return false;
			if (key == cur->key)
				return true;
			else if (key < cur->key)
				return contain(cur->lson, key);
			else // key > cur->key
				return contain(cur->rson, key);
		}

		static void preOrder (TNode *cur) {
			if (cur == NULL) return;
			cout << setw(4) << cur->key;
			preOrder(cur->lson);
			preOrder(cur->rson);
		}

		static void inOrder (TNode *cur) {
			if (cur == NULL) return;
			inOrder(cur->lson);
			cout << setw(4) << cur->key;
			inOrder(cur->rson);
		}

		static void postOrder (TNode *cur) {
			if (cur == NULL) return;
			postOrder(cur->lson);
			postOrder(cur->rson);
			cout << setw(4) << cur->key;
		}

		static void display (TNode *cur) {
			bool *isRight = new bool[cur->count()];
			Display_main(cur, isRight, 0);
			delete [] isRight;
		}

		static void Display_main (TNode *cur, bool isRight[], int indent) {
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
			cout << setw(4) << cur->key << endl;
			if (cur->lson == NULL && cur->rson == NULL)
				return;

			isRight[indent] = 1;
			Display_main(cur->lson, isRight, indent + 1);
			isRight[indent] = 0;
			Display_main(cur->rson, isRight, indent + 1);
		}

		void clear (TNode *cur) {
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

		void Insert (T key) {
			count++;
			rt = Insert(rt, key);
			rt->fa = rt;
			display();
		}

		void Delete (T key) {
			count--;
			rt = Delete(rt, key);
			rt->fa = rt;
		}

		bool contain (T key) {
			return contain(rt, key);
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
#undef TNode


int main () {
	int n;
	int number;
	cout << left; // 左对齐
	Tree<int> T; // 可使用其他类型的键值对
	scanf("%d", &n);
	for (int i = 0; i < n; i++)  {
		cin >> number;
		T.Insert(number);
	}
	T.display();
	while (1) {
		scanf("%d", &number);
		T.Delete(number);
		T.display();
	}
	return 0;
}
