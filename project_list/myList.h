#ifndef MYLIST_H_
#define MYLIST_H_

#include <cstdio>
#include <iostream>
using namespace std;

// 预先声明，否则在友元声明时会报错
template <typename T> class myList;
template <typename T> class myListnode {
  private:
    myListnode<T> *prev, *next;
    T key;
    // 友元可以访问 private 下的变量
    friend class myList<T>;

  public:
    myListnode(int _key = 0) {
        prev = next = NULL;
        key = T(_key);
    }
    ~myListnode() { prev = next = NULL; }
    bool operator<(const myListnode<T> &e) const { return key < e.key; }
    myListnode<T> &operator=(const myListnode<T> &e) {
        key = e.key;
        return *this;
    }
};

template <typename T> class myList {
// 宏定义 简化写法
#define Node myListnode<T>
  private:
    myListnode<T> *head;

  public:
    myList() {
        head = new Node();
        head->next = head;
        head->prev = head;
    }
    ~myList() {
        if (head->next != head)
            clear();
        delete head;
    }
    myList<T> &operator=(const myList<T> &e) {
        clear();
        Node *idx = head, *cur = e.head->next;
        while (cur != e.head) {
            Node *u = new Node(cur->key);
            head->prev = u; // 尾插法
            u->next = head;
            u->prev = idx;
            idx->next = u;
            idx = u;
            cur = cur->next; // 继续向后遍历
        }
    }
    // 1. 类的控制台
    void setup() {
        char ch;
        int n, value, x;
        int *arr = NULL;
        while (ch != 'q') {
            printf("\33[32m(myList)\033[0m");
            cin >> ch;
            switch (ch) {
            case 'h':
                printf("命令列表：\n");
                printf("1: 用一个数组来初始化链表\n");
                printf("2: 按顺序输出整个链表所有的元素\n");
                printf("3: 插入值为 value 的节点到链表中，作为值为 x "
                       "的结点的后继结点\n");
                printf("4: 删除链表中值为 value 的节点\n");
                printf("5: 判断链表是否为空\n");
                printf("h: 查询命令列表\n");
                printf("d: debug\n");
                printf("c: 删除整个链表\n");
                printf("q: 退出\n");
                break;
            case '1':
                if (!empty())
                    clear();
                scanf("%d", &n);
                arr = new int[n];
                for (int i = 0; i < n; i++)
                    scanf("%d", &arr[i]);
                init(arr, n);
                break;
            case '2':
                try {
                    disp();
                } catch (const char *str) {
                    printf("%s\n", str);
                }
                break;
            case '3':
                scanf("%d%d", &value, &x);
                try {
                    insert_item(value, x);
                } catch (const char *str) {
                    printf("%s\n", str);
                }
                break;
            case '4':
                scanf("%d", &value);
                try {
                    delete_item(value);
                } catch (const char *str) {
                    printf("%s\n", str);
                }
                break;
            case '5':
                printf("%d\n", empty());
                break;
            case 'd':
                dbg();
                break;
            case 'c':
                clear();
                break;
            case 'q':
                break;
            }
        }
        if (arr != NULL)
            delete arr;
    }
    // 2. 调试 dbg
    void dbg() {
        printf("链表头指针地址: %p\n", head);
        printf("当前列表内容: ");
        try {
            disp();
        } catch (const char *str) {
            printf("%s\n", str);
        }
    }
    // 3. 通过数组用尾插法创建链表
    void init(const int arr[], const int &len) {
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
    // a. 插入值为 value 的节点到链表中，作为值为 x 的结点的后继结点
    void insert_item(const int &value, const int &x) {
        Node *idx = head->next;
        while (idx != head) {
            if (idx->key == x && idx->next->key != x) {
                Node *u = new Node(value);
                idx->next->prev = u;
                u->next = idx->next;
                u->prev = idx;
                idx->next = u;
                return;
            }
            idx = idx->next;
        }
        throw "no item with key == x";
    }
    // b. 删除链表中值为 value 的节点
    void delete_item(const int &value) {
        Node *idx = head->next;
        while (idx != head) {
            if (idx->key == value && idx->next->key != value) {
                idx->prev->next = idx->next;
                idx->next->prev = idx->prev;
                delete idx;
                return;
            }
            idx = idx->next;
        }
        throw "item notfound";
    }
    // c. 判断链表是否为空
    bool empty() { return head->next == head; }
    // d. 判断链表中是否有值为 value 的结点
    bool check_item(const int &value) {
        Node *idx = head->next;
        while (idx != head) {
            if (idx->key == value)
                return true;
            idx = idx->next;
        }
        return false;
    }
    // e. 输出整个链表
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
    // f. 删除整个链表
    // 注意删除后仍然可以重新以创建链表，头节点并未被损毁
    void clear() {
        if (empty())
            return;
        Node *idx = head->next;
        while (1) {
            if (idx->prev != head)
                delete idx->prev;
            if (idx == head)
                break;
            idx = idx->next;
        }
        head->next = head;
        head->prev = head;
    }
// 取消宏定义
#undef Node
};

#endif // MYLIST_H_
