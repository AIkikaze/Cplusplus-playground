#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;
const int stack_max_cap = 1e6;

class myStack {
	public:
		int capacity;
		int *base;
		int top;

		explicit myStack (int n = stack_max_cap) {
			top = -1;
			capacity = n;
			base = new int[n];
		}

		myStack& operator=(const myStack &rhs) {
			if (this != &rhs) {
				if (base != NULL) {
					delete [] base;
					base = NULL;
				}
				capacity = rhs.capacity;
				top = rhs.top;
				base = new int[capacity];
				memcpy(base, rhs.base, capacity+1);
			}
			return *this;
		}

		~myStack () {
			delete [] base;
		}
		
		int isEmpty () {
			return top == -1? top : 0; 
		}

		bool push (const int &e) {
			return top < capacity-1? base[++top] = e : false;
		}

		bool pop () {
			return top == -1 ? false : top--;
		}

		bool pop (int &e) {
			return top == -1 ? false : e = base[top--];
		}

		int getTop() {
			return top == -1 ? 0xfffffff : base[top];
		}

		void dispStack () {
			if (top == -1)
				return;
			for (int i = 0; i <= top; i++) {
				printf("[%d]: %d\n", i, base[i]);
			}
		}
};

int main () {
	int n;
	scanf("%d", &n);
	myStack s(2*n);
	for (int i = 0; i < n; i++) {
		int m;
		scanf("%d", &m);
		s.push(m);
	}
	s.dispStack();
	while (1) {
		int m;
		scanf("%d", &n);
		switch(n) {
			case 1:
				scanf("%d", &m);
				s.push(m);
				break;
			case 2:
				s.pop();
				break;
			case 3:
				printf("%d\n", s.getTop());
				break;
			case 4:
				s.dispStack();
				break;
			case 5:
				printf("%d\n", s.isEmpty());
				break;
		}
	}
	return 0;
}
