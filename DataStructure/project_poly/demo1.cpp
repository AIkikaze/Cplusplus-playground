#include <iostream>
#include <cstdio>
#include <typeinfo>
using namespace std;

template <typename T>
class ONE {
public:
	T key;
	ONE (double k = 0): key(k) {  }
	friend ostream& operator << (ostream &out, const ONE<T> &x);
};


ostream& operator <<(ostream &out, const ONE<int> &x) {
	out << "int: " << x.key;
	return out;
}


ostream& operator <<(ostream &out, const ONE<double> &x) {
	out << "double: " << x.key;
	return out;
}

int main () {
	ONE<int> x1(1);
	ONE<double> x2(1.2);
	cout << x1 << endl;
	cout << x2 << endl;
	return 0;
}
