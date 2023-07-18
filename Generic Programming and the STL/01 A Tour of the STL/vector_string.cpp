#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;

int main() {
  vector<string> v;
  string tmp;

  while (getline(cin, tmp)) {
    v.push_back(tmp);
  }

  sort(v.begin(), v.end());
  copy(v.begin(), v.end(), ostream_iterator<string>(cout, "\n"));

  system("pause");
  return 0;
}