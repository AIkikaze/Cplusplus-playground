# 树形结构

## 树的定义

树（tree）是由$n(n \ge 0)$个节点组成的有限集合（记为 $T$ ）。

如果 $n = 0$ ，它是一棵空树，这是树的特例。（没有什么特别的用处，就算是要利用归纳法证明一些树的性质，我们一般也会从n为1或2开始）

如果 $n > 0$ ，这n个节点中有且仅有一个节点作为树的 **根节点** ，简称为 **根** (root)，其余节点可分为 $m(m \ge 0)$ 个互不相交的的有限集 $T_1, T_2, \cdots, T_m$ ，其中每个子集本身又是一颗符合本定义的树，称为根节点的 **子树** (substree)。

注：
1. 这是较为笼统的树的递归定义，不涉及图论的连通性等概念，且默认我们将其视为无向的有根树。
2. 由于定义的递归的，有许多和树有关的数量关系，无法直接从这个定义中看出来。
3. 在图论中无根树的定义一般为不含回路的连通图，有根树的定义为在无根树中取一入度为0且与其余节点互相联通的节点（取法有多种但不唯一），将其作为树根节点，即构成有根树。

树的抽象数据类型表述如下：

ADT Tree{
数据对象：
$D=\left\{a_{i} \mid 1 \leqslant i \leqslant n, n \geqslant 0, a_{i}\right.$ 为 ElemType 类型 $\}$
//ElemType 是自定义类型标识符 数据关系:
$R=\left\{\left\langle a_{i}, a_{j}\right\rangle \mid a_{i}, a_{j} \in D, 1 \leqslant i, j \leqslant n\right.$, 其中有且仅有一个结点没有前驱结点, 其余每个结点 只有一个前驱结点,但可以有零个或多个后继结点 $\}$
基本运算:
InitTree $(\& t)$ : 初始化树, 造一棵空树 $t$.
Destroy Tree( \& $t$ ): 销毁树, 释放树 $t$ 占用的存储空间.
TreeHeight $(t)$ : 求树 $t$ 中的高度.
Parent $(t, p):$ 求树 $t$ 中 $p$ 所指结点的双亲结点.
$\operatorname{Borther}(t, p):$ 求树 $t$ 中 $p$ 所指结点的所有兄弟结点.
Sons $(t, p):$ 求树 $t$ 中 $p$ 所指结点的所有子孙结点.
}

二叉树的类封装：

```c++
struct TreeNode {
  ElemType key; // 节点所存储的数据项
  TreeNode *lson, *rson, *fa; // 三个指针分别指向当前节点的左右子树根节点地址和父节点的地址
};

class Tree {
private:
  TreeNode *root; // 该类所封装树的根节点指针
  int len; // 树中有效节点的个数
public:
  init(TreeNode &rt); // 初始化树，造一棵空树
  clear(TreeNode &rt); // 销毁树，释放树rt占用的内存空间
  getHeight(TreeNode &rt); // 求树的高度
  // ...
};
```

## 树的逻辑表示方法

1. **树形表示法** 简单直接地用标有序号的小圆圈表示节点，用圆圈直接的连线表示节点之间的关系（注意无向树不带箭头）即可。一般地，我们将根节点画在最上方，将树叶节点画在最下方。
2. **维恩图表示法** 由于我们的树是在集合的意义上定义的，我们按树的节点集合与其子树的节点集合之间的包含关系递归地作维恩图即可。
3. **凹入表示法** （条形图表示法）跟用缩进来表示层级关系的大纲视图十分类似
。
4. **括号表示法** 用$A(B_1,B_2,\cdots,B_n)$来表示根节点 $A$ 与根节点的子树 ，进而递归地将树表示出来。（这谁想出来的？我极其怀疑这家伙是不是为了~~节省文本空间~~懒得画图发明的这种表示方法）

## 树的基本术语

1. **节点的度和树的度** 树中某个节点的子树的个数称为该节点的度(degree of node)。树的度(degree of tree)则为其所有节点的度的最大值。一般地，我们将度为m的树称作m次树或者m元树（下文总是使用m元树这一种说法）。
2. **分支节点和叶子节点** 树中度不为0的节点为分支节点，反之度为0的节点为叶子节点。为了简便，我们将度为1的分支节点称为单分支节点，而度为2的节点称为双分支节点。
3. **路径与路径长度** 对于树中任意两个节点 $k_i$ 和 $k_j$ ，由无向树的图论性质，必然存在唯一的从 $k_i$ 到 $k_j$ 的一条通路 $(k_i, k_{i1}, k_{i2}, \cdots, k_{in}, k_j )$ 。而当这条通路上的节点满足：除 $k_1$ 外每一个节点在通路序列中都是前一个节点的后继节点时，我们称 $K_i$ 到 $k_j$ 之间存在一条路径，而 $n+1$ 为路径长度。
4. **孩子节点，父节点和兄弟节点** 若将我们定义的树由上下层级关系转化为图论意义上的有向根树中，那么每一个节点的后继节点称为该节点的孩子节点。（事实上，本文中递归定义的树和图论中的有向根树完全等价。但在树的表示时，我们用无向的连线来表示节点之间的联系，用上下位置的区别来表示前驱和后继的关系，在这一点上与图论意义的树是不同的）。于是，相应的父节点即对应当前的节点的前驱节点，兄弟节点即对应父节点相同的除当前节点外的其他节点。
5 **节点层次和树高** 与节点的度和数的度定义类似，节点的层次指从根节点到当前节点的路径长度，显然根节点的层次为0。而树高指树中所有节点的最大层次。
6. **有序树和无序树** 若树的左右子树是按照一定的次序从左向右安排的，且相对次序不能随意变换，则称为有序树(ordered tree)，否则称为无序树(unordered tree)。一般没有特别说明，默认树都是指有序树。
7. **森林** 若干棵互不相交的树称为森林。

## 树的性质

1. 树中节点数等于所有节点的度数之和加1，即 $n = \sum_{i=1}^{n} deg(v_i) + 1$。更进一步地，有树的节点树等于所有分支节点的度数之和加1。
2. 度为 $m$ 的树在第 $i$ 层上之多有 $m^{i-1}$ 个节点。
3. 高为 $h$ 度数为 $m$ 的树至多有 $\frac{m^h - 1}{m - 1}$ 个节点。
4. 具有 $n$ 个节点 $m$ 元树的最小高度为 $log_m(n(m-1)+1)$。对于二元树，该求树高的公式为 $log_2(n+1)$ 。

## 树的基本运算

1. 寻找满足特定条件的节点，如寻找两个节点的最大公共祖先。
2. 插入或删除某个节点。
3. 遍历树中所有节点。
  3.0 树的遍历一共有三种方式分别是先根遍历，后根遍历和层次遍历。
  3.1 先根便利：先便利根节点，后按照从左到右的顺序递归地遍历根节点的每一棵子树。
  3.2 后根遍历：先按照从左到右的顺序递归地遍历根节点的每一棵子树，再访问根节点。
  3.3 层次遍历：从根节点开始从上到下、从左到右遍历树中的每一个节点。


## 树的存储结构

### 线性表存储

一般选择层次遍历的顺序对树的各个节点进行线性表存储，对于每个节点可以相应的封装指向该节点的孩子节点和父节点的指针，加快遍历效率。若只选择其中之一进行封装，那么节点的另一组节点指标将需要通过遍历整个存储结构来查询。例如，仅封装父节点位置作为节点指针项的线性表在遍历某个节点的孩子节点时，需要遍历整个数组来进行查找。

特别地，如果是二叉树的线性表存储，我们一般称之为线段树。由于其节点指标在二叉树的结构下有很好的对应关系，故无需设置节点的指针项就可以很方便地得出其父子节点的位置。

### 链表存储

因为是链表存储，节点的指标之间不存在逻辑联系，故一般每个节点会全部封装指向其对应的父子节点的指针。特别地，只有在不需要对节点进行向上遍历时，才会考虑仅封装指向起孩子节点的指针。另外，在孩子节点较多时，为了减少空间占用，也会选择对指向孩子节点的指针集也使用链表的形式来存储。

例如：

用线性表存储孩子节点的指针项
```c++
struct TreeNode {
  ElemType data;
  TreeNode *sons[maxSons];
};
```

用链表存储孩子节点的指针项
```c++
struct TreeNode {
  ElemType data;
  TreeNode *bro; // 从左至右指向下一个兄弟节点
}
```

# 二叉树

## 二叉树的定义

二叉树(binary tree) 是一个有限的结点集合, 这个集合或者为空, 或者由一个根结点和两棵互不相交的称为左子树 (left subtree) 和右子树 (right subtree) 的二叉树组成。

注：
1. 注意这里也是递归定义。
2. 在图论的意义上，这里定义的二叉树等价于正则有序二元树。
3. 二叉树严格区分左右子树的次序，一般我们不会说无序的二叉树，而会说无序的二元树。
4. 与二元树的区别：二元树要求节点的最大度数取到2，而二叉树要求节点的度要么为0，要么为2。
5. 二叉树结构简单、存储效率高，其运算算法也相对简单，并且任何 $m$ 元树都可以转化为二叉树。因此，二叉树具有很重要的地位。

在此定义下，我们容易看出二叉树的五种基本形态：

![20211103Gj09gr](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211103Gj09gr.png)

为了方便之后的叙述，这里再添加2个和二叉树有关的术语：

1. **满二叉树(full binary tree)** 所有的分支节点都有左右孩子节点，并且叶子节点都集中在二叉树的最下一层，这样的二叉树称为满二叉树。
2. **层序编号(level coding)** 对于一棵满二叉树可以进行层序编号，从1开始由根节点按从上至下、从左到右的顺序编号。如图：

![20211103RR1x2G](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211103RR1x2G.png)

3. **完全二叉树(complete binary tree)** 二叉树中最多只有最下面两层的接地那度数可以小于2，并且最下面一层的叶子节点都依次排列在该层最左边的位置上，这样的树称为完全二叉树。容易知道，完全二叉树也可以按照相同的规则进行层序编号。

利用好层序编号，我们可以容易地在树节点之间进行遍历。对于编号为 $i$ 的节点，其左右孩子节点编号分别为 $2i, 2i+1$，其父节点的编号为 $\left \lfloor i/2 \right \rfloor$。

## 二叉树与树、森林之间的转换

### 从森林、树转换为二叉树

将单棵树转化为一棵二叉树：
1. 树相邻兄弟节点之间链接一条连线
2. 对树中每一个节点只保留它与长子之间的连线，删除与其他孩子之间的连线。
3. 以树的根节点为轴心将整棵树顺时针旋转 $45^{o}$ ，使之层次分明。

将多棵树构成的森林转化为二叉树：
1. 将森林中的每棵树转化为二叉树
2. 第一棵二叉树不动，从第二棵二叉树开始，依次把后一棵二叉树的根结点作为前一 棵二叉树根结点的右孩子结点，当所有二叉树连在一起后，此时得到的二叉树就是由森林转 换得到的二叉树。

### 二叉树还原为树、森林

二叉树还原为单棵树：
1. 若某结点是其双亲的左孩子，则把该结点的右孩子、右孩子的右孩子等都与该结点 的双亲结点用连线连起来。
2. 删除原二叉树中所有双亲结点与右孩子结点之间的连线。
3. 整理由前面两步得到的树，即以根结点为轴心，逆时针转动 $45^{\circ}$，使之结构层次分明。实际上，二叉树的还原就是将二叉树中的左分支保持不变，将二叉树中的右分支还原成兄弟关系。

二叉树还原为森林：
1. 抺掉二叉树根结点右链上的所有结点之间的“双亲一右孩子”关系, 将其分成若干 个以右链上的结点为根结点的二叉树,设这些二叉树为 $b t_{1} 、 b t_{2} 、 \cdots, b t_{m}$ 。
2. 分别将 $\mathrm{bt}_{1}, \mathrm{bt}_{2}, \cdots, \mathrm{bt}_{m}$ 二叉树各自还原成一棵树。

## 二叉树的存储结构

### 顺序存储结构

将二叉树添加空节点直到恰好构成一棵完全二叉树，然后以层次编号的顺序依次存储在一组地址连续的存储单元中。（注意空节点也要编号）

编号效果如图：

![20211103KUz1Bm](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211103KUz1Bm.png)

线段树的类封装：

```c++
struct lineTree {
  ElemType rt[maxSizes];
  bool isempty[maxSizes];
  int len;
};
```

### 链式存储结构

二叉树的链式存储结构是指用一个链表来存储一棵二叉树。

树的类封装：

```c++
struct BTnode {
  ElemType data;
  BTnode *lson, *rson, *fa;
};
```

# 平衡树(avl tree)

定义：平衡树的全称其实是 **平衡二叉树查找**，是由前苏联的数学家 Adelse-Velskil 和 Landis 在 1962 年提出的高度平衡的二叉树，根据科学家的英文名也称为 AVL 树。它具有如下几个性质：
1. 可以是空树
2. 若不是空树，则左子树和右子树都是平衡树，且两者高度之差的绝对值$\le 1$ 

要维护一棵二叉树的平衡性，我们首先要找到一个指标来对一棵树的平衡性进行刻画。根据定义，不妨设树中每个节点有一数据项为平衡因子，其值等于该节点左右子树高度之差（左-右）。即记为 $balance(v_i) = height(v_i->lson) - height(v_i->rson)$。

由平衡树的性质可知，对于一棵平衡树的每一个节点其平衡因子的取值为 $0,-1,+1$。相对于地，如果在树的操作过程中，某一节点的平衡因子发生变化使得其绝对值大于1， 那么我们知道这时需要调整树的结构保持树的平衡。

调整的基本方式就是左旋和右旋

# B树与B+树

## B树

二叉排序树和平衡二叉树都是用作内查找的数据结构，即被查找的数据集不太，可以放在内存中。而这里的B树和B+树是用作外查找的数据结构，其中的数据存放在外部存储器中。

B树中所有节点的孩子节点的最大值称为B树的阶，从查找效率考虑，一般要求 $n \ge 3$ 。一棵m阶的B树要么是一棵空树，要么是一棵满足下列要求的m元树：

1. 树中每个节点最多有m棵子树（即最多含有m-1个关键字）
2. 若根节点不是叶子节点，则根节点最少有两棵子树
3. 除根节点以外, 所有非叶子节点最少有 $\lceil m / 2\rceil$ 棵子树(即最少含有 $\lceil m / 2\rceil-1$ 个关键字)；
4. 每个节点的结构为：
![202111047HJAR8](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/202111047HJAR8.png)
    其中，$n$ 为该节点中的关键字个数，除根节点以外，其他所有节点的关键字个数 $n$ 满足 $\lceil\mathrm{m} / 2\rceil-1 \leqslant n \leqslant m-1 ; k_{i}(1 \leqslant i \leqslant n)$ 为该节点的关键字且满足 $k_{i}<k_{i+1} ; p_{i}(0 \leqslant i \leqslant n)$ 为该节点的孩子节点指针，满足 $p_{i}(0 \leqslant i \leqslant n-1)$ 所指子树上节点的关键字均大于 $k_{i}$ 且小于 $k_{i+1}$（特别地，$p_{n}$ 所指子树上节点的关键字均大于 $k_{n}$ ，$p_{0}$ 所指子树上节点的关键字均小于 $k_1$）。
5. 所有的外部节点在同一层，且不带信息。
    B树中外部节点实际上并不存在（指向这些节点的指针为空），为了叙述方便，我们也不会在图示中画出外部节点层，而在实际工程运用中，外部节点代表的意义是查找失败的情况。

例如下图中一棵 $m=3$ 的B树，它满足：
1. 每个节点的孩子节点的个数小于等于3；
2. 除根节点外其他节点至少有 $\lceil m/2 \rceil = 2$  个孩子；
3. 根节点有2个孩子节点
4. 分支节点的关键字个数大于等于 $\lceil m/2 \rceil - 1 = 1$，小于等于 $m-1 = 2$；
5. 所有外部节点在同一层上，树中共有17个关键字，外部节点有18个。

![20211104s06vbG](https://picofwwzy-1307689287.cos.ap-shanghai.myqcloud.com/uPic/20211104s06vbG.png)

B树中，节点类型的声明如下：
```c++
#define MAXM 10
typedef int ElemType;
struct BTnode {
  int keynum;              // 节点当前拥有的关键字个数
  ElemType key[MAXM];      // 关键字序列,key[0]不可用
  BTnode *fa, *sons[MAXM]; // 父节点指针和孩子节点指针
};
int Max, Min;              // 每个节点最多和最少关键字个数
int m;                     // B树的阶数
```

## B树的基本操作

1. B树的查找
2. B树的插入
    分两种情况：
    2.1 节点关键字个数小于 Max
    2.2 节点关键字个数等于 Max
3. B树的删除
    分三种情况：
    3.1 节点关键字个数大于 Min
    3.2 节点关键字个数等于 Min 且 左右兄弟节点关键字个数大于 Min
    3.3 节点关键字个数等于 Min 且 左右兄弟节点关键字个数等于 Min

## B+树


# 红黑树(red-black tree)

红黑树是每个节点都带有颜色属性的二叉树，每个节点的颜色要么为红色，要么为黑色。在二叉查找树强制一般要求之外，对于任何有效的红黑树我们增加了如下的额外要求：

1. 根节点是黑色的
2. 节点要么是黑的，要么是红的
3. 所有叶子节点为黑色（叶子节点是最后的null节点）
4. 每个红色节点的两个子节点都是黑色（从根节点到每个叶子节点的所有路径上不能出现两个连续的红色节点）
5. 从任一节点到每个叶子节点的所有路径都包含相同数目的黑色节点

这些约束强制了红黑树的关键性质：从根到叶子节点的最长的可能路径（红黑交替）不多于最短的可能路径（全是黑色）的两倍长。从结果上来说，这使得这棵树大致是平衡的。
