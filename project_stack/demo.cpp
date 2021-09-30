/* title: 前缀表达式转中缀表达式 */
/* class: 模板栈 */
/* example: */ 
/* 	in: 8-(3+2*6)/5+4 */
/* 	out: 8 3 2 6 * + 5 / - 4 + */
/* comments: 修改pexp的时候添加了'x'和'y'作为数字和运算符的标识，原先是需要在写另一个题目的时候用到这个的，因为可以在扫描字符串的时候给数组和运算符定位。但是后来想了想还是用队列和栈做更加方便（写在另一个practice里），这里的修改索性就留了下来。 */
#include <iostream>
#include <cstdio>
using namespace std;
const int stack_max_capacity = 1e2;

template<typename T>
inline void dbg (T const& elem) {
	cout<<"dbg: "<<elem<<endl;
}

template<typename T>
class myStack {
	private:
		T *base;
		int top;
	public:
		
		myStack (int capacity = stack_max_capacity) {
			base = new T[capacity];
			top = -1;
		}

		~myStack () {
			delete [] base;
			top = -1;
		}
		bool empty () const{	
			return top == -1? top : false;
		}

		void push (T const& elem) {
			base[++top] = elem;
		}

		void pop () {
			if (!empty()) top--;
		}

		T getTop ()const {
			if (empty()) {
				throw "out of range";
			}
			return base[top];
		}
};

void PostExp (char *exp, char *pexp) {
	myStack<char> s;
	char tmp;

	while (*exp) {
		if (*exp >= '0' && *exp <= '9') {
			*pexp++ = 'x';
			while (*exp >= '0' && *exp <= '9')
				*pexp++ = *exp++;
			*pexp++ = 'x';
		}
		if (*exp == '(') {
			s.push(*exp);
		}
		if (*exp == ')') {
			// 如果栈中有'('我们需要把'('从栈中弹出
			for (tmp = s.getTop(), s.pop(); tmp != '(';) {
				*pexp++ = 'y';
				*pexp++ = tmp;
				*pexp++ = 'y';
				tmp = s.getTop();
				s.pop();
			}
		}
		if (*exp == '+' || *exp == '-') {
			if (!s.empty()) {
				for (tmp = s.getTop(); tmp != '('; ) {
					*pexp++ = 'y';
					*pexp++ = tmp;
					*pexp++ = 'y';
					s.pop();
					if (s.empty()) break;
					tmp = s.getTop();
				}
			}
			s.push(*exp);
		}
		if (*exp == '*' || *exp == '/') {
			if (!s.empty()) {
				for (tmp = s.getTop(); tmp == '*' || tmp == '/';) {
					*pexp++ = 'y';
					*pexp++ = tmp;
					*pexp++ = 'y';
					s.pop();
					if (s.empty()) break;
					tmp = s.getTop();
				}
			}
			s.push(*exp);
		}
		exp++;
	}
	while (!s.empty() && (tmp = s.getTop()) ) {
		*pexp++ = 'y';
		*pexp++ = tmp;
		*pexp++ = 'y';
		s.pop();
	}
}

void dispExp (char *exp) {
	for (int i = 0; i < strlen(exp); i++) {
		if (!i && exp[i] == 'x') continue;
		if (exp[i] != 'x' && exp[i] != 'y') printf("%c", exp[i]);
		else {
			printf(" ");
			while (exp[i+1] == 'x' || exp[i+1] == 'y')
				i++;
		}
	}
	printf("\n");
}

int main () {
	printf("give me a infex expression without and <space>\n");
	char _exp[100], _pexp[100];
	scanf("%s", _exp);
	PostExp(_exp, _pexp);
	dispExp(_pexp);
	return 0;
}
