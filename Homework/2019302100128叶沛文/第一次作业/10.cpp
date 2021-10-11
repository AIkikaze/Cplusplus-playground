/*
 * author: wenwenziy
 * title: 中缀转后缀并计算10
 * last edited: 2021-10-10
 * class: 数组模拟栈
 */
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cmath>
using namespace std;
const int MAX_LEN = 1e6;
char st[MAX_LEN], _exp[MAX_LEN];
int top = -1;

int lev (const char &ch) {
	switch (ch) {
		case '+':
		case '-':
			return 1;
		case '*':
		case '/':
			return 2;
		case '^':
			return 3;
		case ')':
		case '(':
			return -1;   // 注意()的优先级
	}
	return 0;
}

char* infix2posfix (char *ch) {
	int len = strlen(ch), j = 0;
	char *cs = new char[len<<1];
	// 注意始终维护栈中的算符 lev 为递增
	// 例如 st: + * 最终出栈为 * +
	// 而 st - * + 则要在 + 入栈前优先将栈内
	// 优先级高的算符弹出
	for (int i = 0; i < len; i++) {
		if (!lev(ch[i])) {
			while (i < len && !lev(ch[i])) cs[j++] = ch[i++];
			cs[j++] = ' ';
		}
		if (lev(ch[i]) == -1) {
			if (ch[i] == '(') st[++top] = ch[i];  // '(' 优先入栈
			else {
				while (top > -1 && st[top] != '(')  // 处理()内的运算符
					cs[j++] = st[top--];
				top--;   // 将(弹出
			} 
		}
		// 注意 2^3^2 同级不能优先运算 其正确的运算顺序为 2^(3^2)
		if (lev(ch[i]) == 3) st[++top] = ch[i];
		if (lev(ch[i]) == 2) {
			// 这里取 = 意味着同级算符优先算入栈的先算
			// 对于 * / 来说其实不影响结果
			while (top > -1 && lev(st[top]) >= 2) cs[j++] = st[top--];
			st[++top] = ch[i];
		}
		if (lev(ch[i]) == 1) {
			// 注意这里必须取 =
			while (top > -1 && lev(st[top]) >= 1) cs[j++] = st[top--];
			st[++top] = ch[i];
		}
	}
	while (top > -1) cs[j++] = st[top--];
	return cs;
}

int solveExp (char *ch) {
	int *_st = new int[strlen(ch)>>1];
	int _top = -1;
	for (int i = 0; i < strlen(ch); i++) {
		if (!lev(ch[i])) {
			int tmp = 0, t = i;
			while (t < strlen(ch) && ch[t] != ' ') {
				tmp *= 10;
				tmp += ch[t++]^48;
			}
			_st[++_top] = tmp;
			i = t;
		}
		if (lev(ch[i]) > 0) {
			/* for (int j = 0; j < _top+1; j++) */
			/* 	printf("%d ", _st[j]); */
			/* printf("\n"); */
			int tmp = 0;
			switch (ch[i]) {
				case '+': tmp = _st[_top-1] + _st[_top]; break;
				case '-': tmp = _st[_top-1] - _st[_top]; break;
				case '*': tmp = _st[_top-1] * _st[_top]; break;
				case '/': tmp = _st[_top-1] / _st[_top]; break;
				case '^': tmp = pow(_st[_top-1], _st[_top]); break;
			}
			_st[--_top] = tmp;
			/* for (int j = 0; j < _top+1; j++) */
			/* 	printf("%d ", _st[j]); */
			/* printf("\n当前位置：[%d] %d\n", i, tmp); */
			/* getchar(); */
		}
	}
	_top = _st[_top];
	delete [] _st;
	return _top;
}

int main () {
	scanf("%s", _exp);
	char *pexp = infix2posfix (_exp);
	printf("%s\n", pexp);
	printf("%d\n", solveExp(pexp));
	delete [] pexp;
	return 0;
}
