#include <algorithm>
#include <iostream>
#include <iterator>
#include <concepts>
#include <vector>
#include <cstdlib>
#include <cstdio>
using namespace std;

template <typename T>
struct node {
  using value_type = T;  // value_type should be the same as T

  T val;
  node* next;  // next should be a pointer to the same Node type

  node(const T& value = T(), node* n = nullptr) : val(value), next(n) {}
};

// NodeConcept concept to constrain the generic type
template <typename T>
concept NodeConcept = requires(T node) {
  // typename T::value_type;
  std::is_same_v<decltype(node.val), typename T::value_type>;
  std::is_same_v<decltype(node.next), T*>;
};

static_assert(NodeConcept<node<int>>, "type error!\n");

/// @brief input_iterator 适配器
/// @tparam Node 
template <NodeConcept Node>
struct node_wrap {
  /// @brief 必要的声明
  using value_type = Node;
  using reference = const Node&;
  using pointer = const Node*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;

  Node* ptr;

  node_wrap(Node* p = nullptr) : ptr(p) {}
  reference operator*() const { return *ptr; }
  pointer operator->() const { return ptr; }

  const node_wrap& operator++() { ptr = ptr->next; return *this; }
  node_wrap operator++(int) { node_wrap tmp = *this; ++*this; return tmp; }

  bool operator==(const node_wrap& i) const { return ptr == i.ptr; }
  bool operator!=(const node_wrap& i) const { return ptr != i.ptr; }
};

template <typename T>
requires std::equality_comparable<T>
bool operator==(const node<T>& x, T val) {
  return x.val == val;
}

int main() {
  using Node = node<int>;

  vector<int> data = {5, 4, 1, 3, 9};
  Node* head = new Node();
  Node* cur = head;
  for (auto& val : data) {
    cur->val = val;
    if (val != data.back()) {  // Check if it's not the last element
      cur->next = new Node();
      cur = cur->next;
    }
  }
  cur->next = nullptr;  // Set the next pointer of the last node to nullptr

  // 利用适配器, 我们可以对自定义链表使用 std::find 函数
  int key = 9;
  auto result = find(node_wrap<Node>(head), node_wrap<Node>(), key);

  if (result != node_wrap<Node>()) {
    int dist = distance(node_wrap<Node>(head), result);
    printf("The result is list[%d] = %d\n", dist, result->val);
  } else {
    cout << "Not Found!" << endl;
  }

  // 利用适配器, 我们可以使用 auto 关键字来遍历链表
  cout << "The linked list is : " << endl;
  for (auto iter = node_wrap<Node>(head); iter != node_wrap<Node>(); iter++) {
    cout << iter->val << " ";
  }
  cout << endl;

  system("pause");

  for (auto iter = head; iter != nullptr;) {
    auto tmp = iter->next;
    delete iter;
    iter = tmp;
  }
  return 0;
}