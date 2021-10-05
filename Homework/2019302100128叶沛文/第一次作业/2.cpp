/* 
 * author: wenwenziy
 * title: 双向循环链表实现2-a,b,c,d,e,f
 * class: myList, myListnode
 * last edited: 2021-10-5
 * functions and features: 
 * warning: 
 */
#include <iostream>
#include <cstdio>
#include <system_error>
using namespace std;

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
    void setup () {}
    //2. 调试 dbg
    void dbg () {
      printf("链表头指针地址: %ld\n", head);
      printf("当前列表内容: ");
      disp();
    }
    //3. 通过数组用尾插法创建链表
    void init (const int arr[], int len) {
      Node *idx = head, *u;
      for (int i = 0; i < len; i++) {
        u = new Node(arr[len]);
        head->prev = u;
        u->next = head;
        u->prev = idx;
        idx->next = u;
        idx = u;
      }

    }
    //a. 插入值为 value 的节点到链表中，作为值为 x 的结点的后继结点
    void insert_item (const int &value, const int &x) {
      Node *idx = head;
      while (idx->next != head) {
        if (idx->key == x && idx->next->key != x) {
          Node *u = new Node(value);
          idx->next->prev = u;
          u->next = idx->next;
          u->prev = idx;
          idx->next = u;
          break;
        }
        idx = idx->next;
      }
    }
    //b. 删除链表中值为 value 的节点
    void delete_item (const int &value) {
      Node *idx = head;
      while (idx->next != head) {
        if (idx->key == value && idx->next->key != value) {
          idx->prev->next = idx->next;
          idx->next->prev = idx->prev;
          delete idx;
          break;
        }
        idx = idx->next;
      }
    }
    //c. 判断链表是否为空
    bool empty() {
      return head->next == head;
    }
    //d. 判断链表中是否有值为 value 的结点
    bool check_item (const int &value) {
      Node *idx = head;
      while (idx->next != head) {
        if (idx->key == value)
          return true;
        idx = idx->next;
      }
      return false;
    }
    //e. 输出整个链表
    void disp() {
      if (!empty()) 
        throw "list is empty";
      Node *idx = head;
      while (idx->next != head) {
        cout << idx->key << ' ';
        idx = idx->next;
      }
      cout << endl;
    }
    //f. 删除整个链表
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
#undef Node
};

int main () {

  return 0;
}
