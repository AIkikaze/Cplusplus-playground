/*
 * author: wenwenziy
 * title: 实现迎面增长栈4-入栈,出栈,取栈顶,判断是否空栈
 * last edited: 2021-10-08
 * class: shareStack
 * functions: 见注释
 * features: 
 * 1. 之前常用的memcpy函数替换为了 for循环的矮个赋值(拷贝)
 * 可以避免预期之外指针被拷贝的情况(比如T为一个含指针的结构体)
 * 2. 可以使用数组来重构其中一个栈,在数组长度超过当前共享栈容量时,
 * 可以实现共享栈容量的动态增长,具体见ini(). 另外要在程序中检验这
 * 个特性,需要先手动初始化一个容量较小的栈(否则默认为最大容量1e7),
 * 然后再用一个较长的数组去初始化共享栈
 * warnings: 
 * 1. shareStack中包含两个栈共享同一段内存,在相关函数实现时
 * 代码写得比较精简,回顾时要多注意
 * 2. 特性2共享栈的动态增长仅在用数组去初始化的时候实现,在元素入栈
 * 时仅会提醒"当期栈容量不足"的信息
 */
#include <iostream>
#include <cstdio>
using namespace std;
const int max_stack_cap = 1e7; 

template<typename T>
class shareStack {
	private:
		int capacity, top[2];
		T *base;
	public:
		// 带explicit关键字的构造函数
		explicit shareStack (int cap = max_stack_cap) {
			capacity = cap;
			top[0] = -1;
			top[1] = capacity;
			base = new T[cap];
		}
		// 析构函数
		~shareStack () {
			if (base != NULL) delete [] base;
			base = NULL;
		}
		// 重载拷贝函数
		shareStack& operator= (const shareStack &rhs) { 
			if (this != &rhs) {
				if(base != NULL) clear();
				capacity = rhs.capacity;
				top[0] = rhs.top[0];
				top[1] = rhs.top[1];
				base = new T[capacity];
				for (int i = 0; i < capacity; i++)
					base[i] = rhs.base[i];
			}
			return *this;
		}
		// 调试函数
		void dbg () {
			printf("当前类地址: %p\n", this);
			printf("当前base起始地址: %p\n", base);
			printf("当前栈容量: %d\n", capacity);
			printf("当前栈顶位置: %d %d\n", top[0], top[1]);
			printf("当前正向栈中元素: \n");
			for (int i = 0; i <= top[0]; i++) 
				printf("%d%c", base[i], i==top[0]? '\n':' ');
			printf("当前逆向栈中元素: \n");
			for (int i = capacity-1; i >= top[1]; i--)
				printf("%d%c", base[i], i==top[1]? '\n':' ');
		}
		// 类的控制台
		void setup () {
			char ch;
			int n, value, ori;
			while (ch != 'q') {
				printf("\33[32m(shareStack)\033[0m");
				cin >> ch;
				switch (ch) {
					case 'h':
						printf("命令列表：\n");
						printf("0: 创建一个容量为n的空栈\n");
						printf("1: 用长度为n的一个数组来"
									 "初始化ori方向的栈\n");
						printf("2: 将值为value的节点以ori方向入栈\n");
						printf("3: 按ori方向出栈\n");
						printf("4: 输出ori朝向的栈顶元素\n");
						printf("h: 查询命令列表\n");
						printf("d: debug\n");
						printf("c: 清空整个栈\n");
						printf("q: 退出\n");
						break;
					case '0':
						scanf("%d", &n);
						*this = shareStack(n);
						break;
					case '1':
						scanf("%d%d", &n, &ori);
						try {
							dbg();
							init(n, ori);
							dbg();
						} catch (const char *str) {
							printf("%s\n", str);
							break;
						}
						for (int i = 0; i < n; i++) 
							scanf("%d", &value), push(value, ori);
						break;
					case '2':
						scanf("%d%d", &value, &ori);
						if(!isfull()) push(value, ori);
						else printf("当前栈容量不足\n");
						break;
					case '3':
						scanf("%d", &ori);
						pop(ori);
						break;
					case '4':
						scanf("%d", &ori);
						if (getTop(ori) != 0xfffffff) printf("%d\n", getTop(ori));
						else printf("当前朝向栈为空\n");
						break;
					case 'd':
						dbg();
						break;
					case 'c':
						clear(); // 默认清空整个shareStack
						break;
					case 'q':
						break;
				}
			}
		}
		// ori方向上栈的初始化,申请一段长为len的空间
		// 如果当前的capacity不满足要求,则将其扩充为原来的两倍
		void init (const int &len, const int &ori) {
			// 计算另一个栈当前的长度
			int _len = !ori? capacity-top[1] : top[0]+1;
			if (len+_len > capacity && capacity == max_stack_cap) 
				throw "当前栈容量已达最大上界";
			if (len+_len < capacity) {
				clear(ori);
				return;
			}
			int _cap = 1, _ori = !ori;
			T *_base;
			if (capacity<<1 > max_stack_cap && len+_len < max_stack_cap)
				_cap = max_stack_cap;
			else 
				while (len+_len >= _cap)
					_cap <<= 1;
			_base = new T[_cap];
			if (!isempty(_ori)) { // 将另一个栈的内容复制过来
				int j = _ori*(_cap+1)-1;
				for (int i = _ori*capacity; ; i-=(_ori<<1)-1) {
					j-=(_ori<<1)-1;
					_base[j] = base[i];
					if (i == top[_ori]) break;
				}
				top[_ori] = j;
			}
			else top[_ori] = _ori*(_cap+1)-1;
			top[ori] = ori*(_cap+1)-1;
			base = _base;
			capacity = _cap;
		}
		// 判断是否满栈
		bool isfull () {
			return top[0]+1 == top[1];
		}
		// 判断是否空栈
		bool isempty (const int &ori) {
			return top[ori] == ori-1+ori*capacity;
		}
		// 入栈 e为入栈元素 ori代表栈的朝向
		void push (const T &e, const int &ori) {
			if (!isfull()) {
				if (ori != 0 && ori != 1) return;
				base[top[ori]-(ori<<1)+1] = e;
				top[ori] -= (ori<<1)-1;
			}
		} 
		// 出栈 ori表示栈的朝向
		void pop (const int &ori) {
			if (!isempty(ori)) {
				if (ori != 0 && ori != 1) return;
				top[ori] += (ori<<1)-1;
			}
		}
		// 取栈顶 
		T getTop (const int &ori) {
			if (!isempty(ori))
				return base[top[ori]];
			else return 0xfffffff;
		}
		// 清空ori方向的栈,如果ori缺省或者为2则全部清空
		void clear (int ori = 2) {
			if (ori == 2 && base != NULL) {
				capacity = 0;
				top[0] = -1;
				top[1] = capacity;
				delete [] base;
				base = NULL;
			}
			else 
				top[ori] = ori*(capacity+1)-1;
		}
};

int main () {
	shareStack<int> S;
	S.setup();
	return 0;
}
