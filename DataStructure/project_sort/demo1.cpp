#include <_ctype.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <ctime>
using namespace std;
const int arrMax = 100;

namespace sort_method {
  // 经典冒泡排序
  void bubble_sort (int arr[], int len) {
    for (int i = 0; i < len-1; i++) 
      for (int j = 0; j < len-i-1; j++) 
        if (arr[j] > arr[j+1])
          swap(arr[j], arr[j+1]);
    for (int i = 0; i < len; i++) 
      printf("%d%c", arr[i], i!=len-1? ' ':'\n');
  }

  // 改进1：记录上一次交换的位置，减少比较次数
  void ex_bubble_sort1 (int arr[], int len) {
    int laswap = len-1, tmp;
    while(1) {
      tmp = laswap;
      for (int j = 0; j < tmp; j++) 
        if (arr[j] > arr[j+1])
          swap(arr[j], arr[j+1]), laswap = j;
      if (tmp == laswap) break;
      else tmp = laswap;
    }
  }

  // 改进2：使用递减的步长来比较，减少前期大范围扫描的比较次数
  void ex_bubble_sort2 (int arr[], int len) {
    int gap = len-1;
    while (1) {
      for (int j = 0; j < len-gap; j++) 
        if (arr[j] > arr[j+gap])
          swap(arr[j], arr[j+gap]);
      gap = (int)(gap/1.4);
      if (!gap) break;
    }
  }

  // 经典选择排序
  void select_sort (int arr[], int len) {
    int minIndex;
    for (int i = 0; i < len-1; i++) {
      minIndex = i;
      for (int j = i+1; j < len; j++) 
        if (arr[j] < arr[minIndex]) 
          minIndex = j;
      if (i != minIndex)
        swap(arr[i], arr[minIndex]);
      printf("[%d]:", minIndex);
      for (int i = 0; i < len; i++) {
        printf("%d%c", arr[i], i!=len-1? ' ':'\n');
      }
    }
  }

  template<typename T>
    void dispHeap (T *heap, T *end) {
      for (int i = 0; i < end-heap; i = (i<<1)+1) {
        printf("heap[%d]:", i);
        for (int j = i; j <= i<<1; j++) 
          printf("%d%c", heap+j<end ? *(heap+j) : -1, j == i<<1 ? '\n' : ' ');
      }
    }

  // 选择排序进阶：堆排序
#define lson (heap + ((i<<1)+1))
#define rson (heap + ((i+1)<<1))
  template<typename T>
    void heapify (T *heap, T *begin, T *end) {
      T *idx = begin;
      int i = 0;
      //printf("heapify from:%ld\n", begin - heap);
      while (idx < end) {
        i = idx - heap;
        //printf("dispHeap from %d:\n", i);
        //dispHeap(heap, end);
        if(rson < end && *lson < *rson && *idx < *rson)
          swap(*idx, *rson), idx = rson;
        else if(lson < end && *idx < *lson)
          swap(*idx, *lson), idx = lson;
        else break;
        //dispHeap(heap, end);
        //getchar();
      }
    }
#undef lson
#undef rson


  template<typename T>
    void heap_sort (T *front, T *end) {
      int len = end-front;
      for (int i = len/2 - 1; i >= 0; i--)
        heapify(front, front+i, end);
      //printf("heap: ");
      //for (int i = 0; i < len; i++) 
      //  printf("%d%c", *(front+i), i==len-1? '\n' : ' ');
      for (int i = len-1; i > 0; i--) {
        swap(*front, *(front+i));
        heapify(front, front, front+i);
      }
    }


  //  经典插入排序
  void insert_sort (int arr[], int len) {
    int tmp, j;
    for (int i = 1; i < len; i++) {
      tmp = arr[i];
      j = i;
      while (arr[j-1] > tmp && j > 0)
        arr[j] = arr[j-1], j--;
      arr[j] = tmp;
    }
  }

  // 二分插入排序
  void halfinsert_sort (int arr[], int len) {
    int tmp, low, high, mid;
    for (int i = 1; i < len; i++) {
      tmp = arr[i];
      low = 0;
      high = i;
      while (low < high) {
        mid = (low+high)>>1;
        if (arr[mid] > tmp) {
          for (int j = high; j > mid; j--)
            arr[j] = arr[j-1];
          high = mid;
        }
        else low = mid+1;
      }
      arr[low] = tmp;
    }
    for (int i = 0; i < len; i++) 
      printf("%d%c", arr[i], i!=len-1? ' ':'\n');
  }

  // 希尔排序，注意希尔分组之后的排序方法有多种写法，可冒泡可插入可选择
  void shell_sort (int arr[], int len) {
    int h = 1, tmp, j;
    while (h < len/3) h = h*3+1;
    while (h >= 1) {
      for (int i = h; i < len; i++) {
        tmp = arr[i];
        for (j = i; j > 0 && arr[j-h] > tmp; j-=h)
          arr[j] = arr[j-h];
        arr[j] = tmp;
      }
      h = h/3;
    }
  }

  // 归并排序 模板+左闭右开
  template<typename T>
    void merge_sort (T *front, T *end) {
      if (front+1 == end) return;
      int i = 0;
      int len = end-front;
      T *mid = front+len/2;
      merge_sort(front, mid);
      merge_sort(mid, end);
      T *b = new T[len];
      T *lidx = front;
      T *ridx = mid;
      while (lidx < mid && ridx < end) {
        if (*lidx < *ridx) 
          b[i++] = *lidx++;
        else 
          b[i++] = *ridx++;
      }
      while (lidx < mid) b[i++] = *lidx++;
      while (ridx < end) b[i++] = *ridx++;
      for (i = 0; i < len; i++) *(front+i) = b[i];
      delete [] b;
    }

  // 快速排序 模板+左闭右开
  template<typename T>
    void quick_sort (T *front, T *end) {
      if (front+1 >= end) return;
      T key = *front;
      T *lidx = front;
      T *ridx = end-1;
      while (lidx < ridx) {
        while (ridx > lidx && *ridx >= key)
          ridx--;
        *lidx = *ridx;
        while (lidx < ridx && *lidx < key)
          lidx++;
        *ridx = *lidx;
      }
      *lidx = key;
      // 注意中间的key已经放好位置就可以不用在排了
      quick_sort(front, lidx);
      quick_sort(lidx+1, end);
    }
}

int main () {
  srand((unsigned)time(NULL));
  int n;
  scanf("%d", &n);
  int *arr = new int[n];
  int *answer = new int[n];
  for (int i = 0; i < n; i++) 
    answer[i] = arr[i] = rand()%arrMax;
  for (int i = 0; i < n; i++) 
    printf("%d%c", arr[i], i!=n-1? ' ':'\n');
  /* for (int i = 0; i < n; i++) */
  /*   cout << '[' << i << ']' << &arr[i] << endl; */

  sort_method::heap_sort(arr, arr+n);
  for (int i = 0; i < n; i++) 
    printf("%d%c", arr[i], i!=n-1? ' ':'\n');

  sort(answer, answer+n);
  for (int i = 0; i < n; i++) 
    printf("%d%c", answer[i], i!=n-1? ' ':'\n');
  delete [] arr;
  delete [] answer;
  return 0;
}

