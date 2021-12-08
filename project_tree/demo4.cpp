/*
 * author: wenwenziy
 * title: 红黑树的实现（set容器类）
 * features: 
 * 1. 枚举类型 enum 的使用
 * 2. 大量的泛型模板和static方法的使用
 * 3. 可通过 T.display() 在控制台输出整棵树（可视化）
 * 4. 红黑树的插入和删除 （已注释）
 * 5. 容器类内外访问权限权限的设置原则
 *    5.1 变量类型统一设置为私有
 *    5.2 对同一类型都适用的方法设置为私有的静态方法
 *    5.3 不在公有的方法中完成算法的实现，仅作为接口提供必要的参数
 *    5.4 构造、析构、拷贝等重载函数统一为公有
 * 测试数据：
 * 10 2 9 8 19 22 3 4 5 2 5 1
 * 2 2 9 8 1
 */
#include <iostream>
#include <cstdio>
#include <iomanip>
using namespace std;

#define RBTNode Treenode<T>
#define isBLACK(cur) ((!cur)||(cur->color == BLACK))
#define isRED(cur) ((cur)&&(cur->color == RED))

enum RBTColor { RED, BLACK };
template <typename T> class Tree;

template <typename T>
class Treenode {
	private:
		RBTColor color;
		T key;
		int numcount;
		RBTNode *lson, *rson, *fa;
		friend class Tree<T>;

		int count (RBTNode *cur) {
			if (cur == NULL) return 0;
			return count(cur->lson) + count(cur->rson) + 1;
		}

	public:
		Treenode (T _key = 0, RBTColor _color = BLACK) {
			lson = rson = fa = NULL;
			key = _key;
			color = _color;
			numcount = 1;
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
		RBTNode *rt; // 搜索树的根节点
		int count; // 树中的节点计数

		static int height (RBTNode *cur) {
			if (cur == NULL) return -1;
			return ax(height(cur->lson), height(cur->rson)) + 1;
		}

		static void lRotate (RBTNode *&root, RBTNode *cur) {
			RBTNode *child = cur->rson;
			cur->rson = child->lson;
			if (child->lson) child->lson->fa = cur;
			child->fa = cur->fa;
			sonex(root, cur, child);
			child->lson = cur;
			cur->fa = child;
		}

		static void rRotate (RBTNode *&root, RBTNode *cur) {
			RBTNode *child = cur->lson;
			cur->lson = child->rson;
			if (child->rson) child->rson->fa = cur;
			child->fa = cur->fa;
			sonex(root, cur, child);
			child->rson = cur;
			cur->fa = child;
		}

		static void Insert (Tree<T> *Tr, T key) {
			RBTNode *cur = Tr->rt;
			RBTNode *pre = NULL;
			while (cur != NULL) {
				if (key == cur->key) {
					cur->numcount++;
					return;
				}
				else if (key < cur->key)
					pre = cur, cur = cur->lson;
				else
					pre = cur, cur = cur->rson;
			}
			Tr->count++;
			cur = new RBTNode(key, RED);
			cur->fa = pre;
			if (pre == NULL)
				Tr->rt = cur;
			else if (key < pre->key)
				pre->lson = cur;
			else
				pre->rson = cur;

			InsertAdjust(Tr->rt, cur);
		}

		static void InsertAdjust (RBTNode *&root, RBTNode *cur) {
			if (root == cur) {
				cur->color = BLACK;
				return;
			}

			RBTNode *par = cur->fa;
			if (isBLACK(par)) {
				return; // 无需调整
			}

			RBTNode *grapar = NULL, *uncle = NULL;
			// 父节点是红色说明父节点不可能为根节点
			for (;isRED(par); par = cur->fa) {
				grapar = par->fa;
				uncle = grapar->lson == par ? grapar->rson : grapar->lson;
				// 父节点为红色，叔叔节点为红色
				if (isRED(uncle)) {
					par->color = BLACK;
					uncle->color = BLACK;
					grapar->color = RED;
					cur = grapar; // 继续处理祖父节点
				}
				// 父节点为红色，叔叔节点为黑色
				else if (par == grapar->lson) {
					if (par->rson == cur) {// LR情况1
						lRotate(root, par); // 左旋后 cur 成为 par 的父节点
						swap(par, cur);
					}// LL情况2
					par->color = BLACK;
					grapar->color = RED;
					rRotate(root, grapar);
				}
				else {
					if (par->lson == cur) {// RL情况3
						rRotate(root, par);
						swap(par, cur);
					}// RR情况4
					par->color = BLACK;
					grapar->color = RED;
					lRotate(root, grapar);
				}
			}
			root->color = BLACK;
		}

		static void sonex (RBTNode *&root, RBTNode *cur, RBTNode *target) {
			if (cur->fa)
				if (cur->fa->lson == cur)
					cur->fa->lson = target;
				else
					cur->fa->rson = target;
			else
				root = target;
		}

		static void Delete (Tree<T> *Tr, T key) {
			RBTNode *cur = Tr->rt, *child = NULL, *par = NULL;
			RBTColor delcolor = BLACK;
			while (cur != NULL) {
				if (key == cur->key) {
					cur->numcount--;
					if (cur->numcount) // >=1
						return;
					else // == 0 准备删除节点
						break;
					// 如果是 set 容器，无需调整 numcount 直接删除整个节点即可
				}
				else if (key < cur->key)
					cur = cur->lson;
				else
					cur = cur->rson;
			}
			if (cur == NULL) return;
			Tr->count--;
			if (cur->rson) {  // cur寻找前序遍历的后缀
				RBTNode *tmp = cur->rson;
				while (tmp->lson) tmp = tmp->lson;
				// 将 tmp 复制到 cur 处，注意不要复制颜色
				cur->key = tmp->key;
				cur->numcount = tmp->numcount;
				// 删除 tmp 处的节点 由于没有左子树，用右子树替换即可
				sonex(Tr->rt, tmp, tmp->rson); // 注意 tmp->rson 可能为 NULL
				par = tmp->fa;
				child = tmp->rson; // child 记录顶替 tmp 的新节点位置
				delcolor = tmp->color;
				delete tmp;
			}
			else { // cur 无右子树
				sonex(Tr->rt, cur, cur->lson); // 注意 cur->lson 可能为 NULL
				par = cur->fa;
				child = cur->lson;
				delcolor = cur->color;
				delete cur;
			}
			// child为红-_节点
			if (delcolor == RED || isRED(child)) {
				if(child) child->color = BLACK; // child 可能为 NULL
			}
			else // child 为黑-黑节点
				DeleteAdjust(Tr->rt, child, par);
		}

		static void DeleteAdjust (RBTNode *&root, RBTNode *x, RBTNode *par) {
			if (x == root)  // 如果是根节点，黑高-1 不影响平衡性
				return;
			RBTNode *bro = par->lson == x ? par->rson : par->lson;
			if (isBLACK(bro)) { // 兄弟节点为黑色
				if (isBLACK(bro->lson) && isBLACK(bro->rson)) { // 兄弟节点的孩子全为黑
					if (isBLACK(par)) DeleteAdjust(root, par, par->fa); // 父节点也为黑 递归地处理 par （此时为双黑）
					else {  // 父节点为红色
						par->color = BLACK;
						bro->color = RED;
						return;
					}
				}
				if (bro == par->lson) { // 兄弟节点的孩子中有红色节点
					if (isRED(bro->lson)) { // LL 红节点为兄弟节点的左子树 （包括双红的情况）
						bro->lson->color = bro->color;
						bro->color = par->color;
						rRotate(root, par);
						par->color = BLACK;
					}
					else { // LR 红节点为兄弟节点的右子树
						bro->rson->color = par->color;
						lRotate(root, bro);
						rRotate(root, par);
						par->color = BLACK;
					}
				}
				if (bro == par->rson) { 
					if (isRED(bro->rson)) { // RR 红节点为兄弟节点的右子树（包括双红的情况）
						bro->rson->color = bro->color;
						bro->color = par->color;
						lRotate(root, par);
						par->color = BLACK;
					}
					else { // RL 红节点为兄弟节点的左子树
						bro->lson->color = par->color;
						rRotate(root, bro);
						lRotate(root, par);
						par->color = BLACK;
					}
				}
			}
			else { // 兄弟节点为红色，则父节点必为黑色
				if (bro == par->lson) { // 左兄弟
					// 右旋 使得 bro 成为 par 子树的根，par 此时为 bro->rson
					// par->lson 为原先的 bro->rson （黑色）
					rRotate(root, par); 
					par->color = RED; 
					bro->color = BLACK;
					DeleteAdjust(root, x, par); // 此时 x 的兄弟节点为 par->lson 属于前面已经处理过的情况
				}
				if (bro == par->rson) { // 右兄弟
					// 左旋 使得 bro 成为 par 子树的根，par 此时为 bro->lson
					// par->rson 为原先的 bro->lson （黑色）
					lRotate(root, par);
					par->color = RED;
					bro->color = BLACK;
					DeleteAdjust(root, x, par); // 此时 x 的兄弟节点为 par->rson 黑色
				}
			}
		}

		// 查找值为 key 是否为树中的元素
		static bool contain (RBTNode *cur, T key) {
			if (cur == NULL)
				return false;
			if (key == cur->key)
				return true;
			else if (key < cur->key)
				return contain(cur->lson, key);
			else // key > cur->key
				return contain(cur->rson, key);
		}

		// 前序遍历
		static void preOrder (RBTNode *cur) {
			if (cur == NULL) return;
			cout << setw(4) << cur->key;
			preOrder(cur->lson);
			preOrder(cur->rson);
		}

		// 中序遍历
		static void inOrder (RBTNode *cur) {
			if (cur == NULL) return;
			inOrder(cur->lson);
			cout << setw(4) << cur->key;
			inOrder(cur->rson);
		}

		// 后序遍历
		static void postOrder (RBTNode *cur) {
			if (cur == NULL) return;
			postOrder(cur->lson);
			postOrder(cur->rson);
			cout << setw(4) << cur->key;
		}

		// 求以 cur 为的根的子树树高
		static int getHeight (RBTNode *cur) {
			return !cur ? -1 : 1 + max( getHeight(cur->lson), getHeight(cur->rson) );
		}

		// 求子树中的最大值
		static int getMax (RBTNode *cur) {
			while (cur->rson) cur = cur->rson;
			return cur->key;
		}

		// 求子树中的最小值
		static int getMin (RBTNode *cur) {
			while (cur->lson) cur = cur->lson;
			return cur->key;
		}

		// 二叉树可视化的初始参数
		static void display (RBTNode *cur) {
			bool *isRight = new bool[cur->count()];
			Display_main(cur, isRight, 0);
			delete [] isRight;
		}

		// 递归地输出二叉树的在控制台的可视化形式
		static void Display_main (RBTNode *cur, bool isRight[], int indent) {
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
			// 在此处可以修改希望输出的节点值，这里仅输出节点的 key 值和颜色 color （仅当 color == RED 时)
			cout << setw(4) << cur->key;
			if (isRED(cur)) cout << "RED" << endl;
			else cout << endl;

			isRight[indent] = 1;
			Display_main(cur->lson, isRight, indent + 1);
			isRight[indent] = 0;
			Display_main(cur->rson, isRight, indent + 1);
		}

		// 释放整棵树的内存，仅在析构时被调用
		void clear (RBTNode *cur) {
			if (cur == NULL) return;
			if (cur->lson == NULL && cur->rson == NULL) {
				delete cur;
				return;
			}
			clear(cur->lson);
			clear(cur->rson);
		}

	// 公共接口
	public:
		Tree () {
			rt = NULL;
			count = 0;
		}
		~Tree () {
			clear(rt);
			count = 0;
		}

		int size () {
			return count;
		}

		bool isEmpty () {
			return count == 0;
		}

		int height () {
			return getHeight(rt);
		}

		int Max () {
			return getMax(rt);
		}
	
		int Min () {
			return getMin(rt);
		}

		void Insert (T key) {
			Insert(this, key);
		}

		void Delete (T key) {
			Delete(this, key);
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
			cout << endl;
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
		T.display();
	}
	// 这里仅测试Delete，其余功能自行测试
	while (1) {
		scanf("%d", &number);
		T.Delete(number);
		T.display();
	}
	return 0;
}

