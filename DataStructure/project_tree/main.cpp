#include <iostream>
#include <cstdio>
using namespace std;

void disp (int *q) {
	printf("%ld\n", sizeof(*q));
}

int main () {
	int a[10]={0,1,2,3,7,8,7,2,0,1};
	int *p = new int[6];
	int *q = a;
	disp (a);
	printf("%ld\n", sizeof a);
	disp (p);
	printf("%ld\n", sizeof *p);
	disp (q);
	printf("%ld\n", sizeof *q);
	return 0;
}
