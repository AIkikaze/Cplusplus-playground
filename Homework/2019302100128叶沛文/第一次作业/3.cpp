/*
 * author: wenwenziy
 * title: 栈结构的实现3-入栈,出栈,取栈顶元素,判断栈是否为空
 * last edited: 2021-10-08
 * class: myStack
 * functions: 见注释
 * features: explicit关键字,重载拷贝函数->见setup()中的
 * *this = myStack(n) 为显式构造+拷贝
 * warnings: 直接使用构造函数(不另外声明init())在对类的重构时
 * behavior 有所不同,注意 *this = myStack(n) 不能写成 myStack(n). 
 * 具体见注释

 */
#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;

template<typename  T>
class myStack {
	private:
		int capacity, top;
		T *base;
	public:
		// 禁用隐式类型转换
		explicit myStack (int cap = 0) {
			top = -1;
			capacity = cap;
			base = new T[cap];
		}
		// 重载拷贝函数
		// 注意在禁用隐式转换后=的右边只能为 myStack 类型
		// 否则为int类型时会调用构造函数进行隐式转换
		myStack& operator=(const myStack &rhs) {
			if (this != &rhs) {
				if (base != NULL) {
					delete [] base;
					base = NULL;
				}
				capacity = rhs.capacity;
				top = rhs.top;
				base = new T[capacity];
				for (int i = 0; i < capacity; i++)
					base[i] = rhs.base[i];
			}
			return *this;
		}
		// 析构函数
		~myStack () {
			if (base != NULL)
				delete [] base;
			base = NULL;
		}
		// 调试函数
		void dbg () {
			printf("当前类地址: %p\n", this);
			printf("当前栈元素起始地址: %p\n", base);
			printf("当前栈容量: %d\n", capacity);
			printf("当前栈高: %d\n", top+1);
			printf("当前栈中元素: \n");
			dispStack();
		}
		// 类的控制台，用于简单展示类的功能
		void setup () {
			char ch;
			int n, value;
			while (ch != 'q') {
				printf("\33[32m(myStack)\033[0m");
				cin >> ch;
				switch (ch) {
					case 'h':
						printf("命令列表：\n");
						printf("0: 创建一个容量为n的空栈\n");
						printf("1: 用一个数组来初始化栈\n");
						printf("2: 按顺序输出整个栈中的元素\n");
						printf("3: 将值为value的节点入栈\n");
						printf("4: 出栈\n");
						printf("5: 输出栈顶元素\n");
						printf("h: 查询命令列表\n");
						printf("d: debug\n");
						printf("c: 清空整个栈\n");
						printf("q: 退出\n");
						break;
					case '0':
						scanf("%d", &n);
						*this = myStack(n);
						break;
					case '1':
						if(!isEmpty()) clear();
						scanf("%d", &n);
						// 注意直接使用 myStack 是无法重新构造当前栈的
						// 会在内存里重新生成一个myStack但由于没有返回指针
						// 它会被遗失在内存中,无法对当前数据项造成改动
						*this = myStack(n<<1);
						for (int i = 0; i < n; i++) 
							scanf("%d", &value), push(value);
						break;
					case '2':
						dispStack();
						break;
					case '3':
						scanf("%d", &value);
						if (!push(value))
							printf("当前栈容量不足\n");
						break;
					case '4':
						if (!pop())
							printf("当前栈为空\n");
						break;
					case '5':
						if ((value = getTop()) && value != 0xfffffff)
							printf("%d\n", value);
						else 
							printf("当前栈为空\n");
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
		}
		// 判断栈是否为空
		int isEmpty () {
			return top == -1? top : 0; 
		}
		// 入栈操作
		bool push (const T &e) {
			return top < capacity-1? base[++top] = e : false;
		}
		// 出栈操作
		bool pop () {
			return top == -1 ? false : top--;
		}
		// 取栈顶元素并出栈
		bool pop (T &e) {
			return top == -1 ? false : e = base[top--];
		}
		// 取栈顶元素
		T getTop() {
			return top == -1 ? 0xfffffff : base[top];
		}
		// 输出栈中所有元素
		void dispStack () {
			if (top == -1) {
				printf("当前栈为空\n");
				return;
			}
			for (int i = 0; i <= top; i++) {
				printf("[%d]: %d\n", i, base[i]);
			}
		}
		// 清空栈
		void clear () {
			if (base != NULL) {
				capacity = 0;
				top = -1;
				delete [] base;
				base = NULL;
			}
		}
};

int main () {
	myStack<int> S;
	S.setup();
	return 0;
}

