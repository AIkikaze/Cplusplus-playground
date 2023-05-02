# 链表的原理和实现

## 链表介绍

线性表的链式存储结构称为链表（linked list）。其中每一个存储结构节点不仅包含元素本身的信息（称为数据域），还包含元素之间的逻辑关系的信息，在 C++中采用指针来实现，称为指针域。

在此基础上，有这样一些术语和定义值得介绍一下：

1. 单链表
   每个节点中除包含数据域之外只设置一个指针域，用于指向后继节点。这样构成的链表称为单向链接表。
2. 双链表
   每个节点除包含数据域外设置两个指针域，分别用于指向前驱节点和后继节点，这样构成的链表称为双向链接表。
3. 空节点
   指数据域不存储信息的节点。对于数据域来说不存储信息意味着元素值为声明时的初始值，对指针来说，意味着指针值为 `NULL`。
4. 头指针，尾指针
   每个链表都有一个头指针和尾指针，分别指向链表的第一个节点（头节点）和最后一个节点（尾节点）。头指针还有一个用处是可以通过头指针可以唯一标识链表。

链表的示意图：

![20211020](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211020.png)

![20211020](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211020.png)

## 简单单向链表

在单链表中，假设每个节点用 linknode 表示，它包括储存元素的数据项 data，以及指向后继节点的指针项 next。于是，我们可以这样来声明一个链表的节点类型 linknode。

```c++
class linknode {
  ElemType data;
  lindnode *next;
};
```

接下来我们要开始思考，如何创建链表中每一个节点，并且把它们链接起来，以及当我们要删去某个节点的时候应该作怎样的操作。以及在代码实现中，我们可能还会考虑一下几个问题：

1. 既然我们要创建链表，那何妨不动态申请存储，使得链表摆脱固定容量的限制。
2. 既然要动态申请内存，那么我们知道利用 C++的 new 和 delete 关键字来向堆中申请内存的时候会调用类的构造、析构函数，这就容易给指针的调用带来问题。
3. 类之间的复制（拷贝）如何作用于其中的数据项 data 和指针项 next？如果我们仅仅想拷贝指针所之内存中的数据，那么就需要重构操作符 '=' 了。

也就是说我们的类声明还需要添加这些内容：

```c++
class linknode {
private:
  ElemType data;
  lindnode *next;
public:
  // 类构造函数
  linknode () {
    ...
  }
  // 类析构函数
  ~linknode () {
    ...
  }
  // 重载拷贝函数
  void operator = (..) {
    ...
  }
};
```

然而这还只是链表的节点，我们还需要再声明一个类用来封装链表的头节点和我们需要的函数操作。

```c++
class myList {
private:
  listnode *head;
public:
  // 链表的基本操作
  // 0. 利用数组初始化链表
  void init(int arr[], int len);
  // 1. 在pos位置处插入值为value的节点
  bool insert(int pos, ElemType value);
  // 2. 删除pos位置处的节点
  bool delete_item(int pos);
  // 3. 判断链表是否为空
  bool empty();
  // 4. 检索链表是否有值为value的节点
  bool check_item(ElemType value)
  // 5. 输出整个链表
  void disp();
  // 6. 清空整个链表
  void clear();
};
```

具体的实现细节已经写在了 `myList.h` 头文件里。这里仅强调一些关键的注意事项：

1. 如何保证两个类之间的访问？
   _答_：（前置知识：[C++访问修饰符](https://www.runoob.com/cplusplus/cpp-class-access-modifiers.html)）注意 `lisnode` 中我们用 `private` 修饰了类的变量成员，这意味着它们在该类的外部是不可访问的。也就是说，我们要在 `myList` 类中对节点的值和指针进行修改和访问是不被允许的（然而这正是我们需要的），然而我们又不想让节点类在除了 `myList` 以外的范围的被修改和访问（这将导致预期意外的链表修改）。于是，我们可以利用友元修饰符，在 `listnode` 中添加如下语句：

   ```c++
   // 添加内容
   class mylist; // 预先声明，防止报错
   // 添加内容结束
   class listnode {
   private:
     ElemType data;
     lindnode *next;
     // 添加内容
     friend class mylist;
     // 添加内容结束
   public:
     ......
   };
   ```

2. 链表遍历的开始和结束条件？
   _答_：利用链表的后继节点指针，我们很容易通过 `p = p->next` 来遍历链表中的节点。对于单向链表，在遍历过程中的开始与 `head` 头节点是否为空有关；结束条件与链表是否循环有关。
   例如，对于头节点为空的循环双向链表，遍历从 `idx = head->next` 开始，到 `idx->next == head` 结束；对于头节点不为空的循环单向链表，遍历从 `idx = head` 开始，到 `idx->next == NULL` 结束。

3. 链表的核心操作：插入和删除
   _答_：尤其需要注意插入删除时，修改节点之间链接指针的顺序：
   1. 对于插入操作，例如在当前节点 `cur` 之后插入 `u` 节点，其操作顺序为：
      ```c++
      u->next = cur->next; // (1)
      cur->next = u;       // (2)
      ```
      保证不出错的原则是：仅在后续操作不依赖当前被赋值元素的值时，对当前元素的赋值才是可行的。否则，请为当前值创建备份或者调整赋值顺序。(这里的操作还比较简单，对于双向的循环链表来说，会更加复杂。)
   2. 对于删除操作，例如删除当前节点 `cur` 之后的 `u` 节点，其操作顺序为：
      ```c++
      cur->next = u->next; // (1)
      delete u;            // (2)
      ```
      注意对 `delete` 所依赖的析构函数的重载
      ```c++
      ~listnode () {
       data = 0;
       next = NULL;
       // 不能 delete next 否则会删除递归地删除 u 的后继节点
      }
      ```
4. 判断链表是否为空也会因为头节点是否为空，在细节上有所不同
   1. 当头节点为空时，判定条件为 `head->next == head`
   2. 当头节点不为空时，判定条件为 `head == NULL`

## 复杂双向循环链表

目前，本项目文件夹中的头文件实现的就是这一类型的链表。具体代码请查看头文件 `myList` 。