/* 
 * author: wenwenziy
 * title: 实现对链表的插入排序
 * last edited: 2021-10-09
 * class myList, myListnode
 */
#include <iostream>
#include <cstdio>
using namespace std;

// 预先声明，否则在友元声明时会报错
template<typename T> class myList;
template<typename T>
class myListnode {
  private:
    myListnode<T> *prev, *next;
    T key;
    // 友元可以访问 private 下的变量
    friend class myList<T>;
  public:
    myListnode (int _key = 0) {
      prev = next = NULL;
      key = T(_key);
    }
    ~myListnode () {
      prev = next = NULL;
    }
    bool operator < (const myListnode<T> &e)const {
      return key < e.key;
    }
    myListnode<T>& operator = (const myListnode<T> &e) {
      key = e.key;
      return *this;
    }
};

template<typename T>
class myList {
// 宏定义 简化写法
#define Node myListnode<T>
  private:
    myListnode<T> *head;
  public:
    myList () {
      head = new Node();
      head->next = head;
      head->prev = head;
    }
    ~myList () {
      if (head->next != head) 
        clear();
      delete head;
    }
    //1. 类的控制台
    void setup ();
    //2. 调试 dbg
    void dbg ();
    //3. 通过数组用尾插法创建链表
    void init (const int arr[], const int &len);
    //a. 插入值为 value 的节点到链表中，作为值为 x 的结点的后继结点
    void insert_item (const int &value, const int &x);
    //b. 删除链表中值为 value 的节点
    void delete_item (const int &value);
    //c. 判断链表是否为空
    bool empty() { return head->next == head; }
    //d. 判断链表中是否有值为 value 的结点
    bool check_item (const int &value);
    //e. 输出整个链表
    void disp() {
      if (empty())
        throw "list is empty";
      Node *idx = head->next;
      while (idx != head) {
        cout << idx->key << ' ';
        idx = idx->next;
      }
      cout << endl;
    }
    //f. 删除整个链表
    //注意删除后仍然可以重新以创建链表，头节点并未被损毁
    void clear() {
      if (empty()) return;
      Node *idx = head->next;
      while (1) {
        if (idx->prev != head)
          delete idx->prev;
        if (idx == head) break;
        idx = idx->next;
      }
      head->next = head;
      head->prev = head;
    }
		// 实现链表的插入排序
		Node *begin () { return head->next; }
		Node *end () { return head; }
		void insert_sort (Node *, Node *);
};

template<typename T>
void myList<T>::init (const int arr[], const int &len) {
	Node *idx = head, *u;
	for (int i = 0; i < len; i++) {
		u = new Node(arr[i]);
		head->prev = u;
		u->next = head;
		u->prev = idx;
		idx->next = u;
		idx = u;
	}
}

template<typename T> 
void myList<T>::insert_sort (Node *begin, Node *end) {
	T tmp;
	Node *idx;
	for (Node *i = begin; i != end; i = i->next) {
		tmp = i->key;
		idx = i;
		while (idx->prev->key > tmp && idx->prev != head) {
			swap(idx->key, idx->prev->key);
			idx = idx->prev;
		}
		idx->key = tmp;
	}
}

// 取消宏定义
#undef Node

int main () {
	myList<int> L;
	int n, *arr;
	scanf("%d", &n);
	arr = new int[n];
	for (int i = 0; i < n; i++)
		scanf("%d", &arr[i]);
	L.init(arr, n);
	printf("排序前: "), L.disp();
	L.insert_sort(L.begin(), L.end());
	printf("排序后: "), L.disp();
	delete [] arr;
	return 0;
}
