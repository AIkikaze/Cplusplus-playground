#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;

typedef vector<char>::iterator strtab_iterator;
typedef pair<strtab_iterator, strtab_iterator> line_iterator;

struct strtab_cmp {
  bool operator()(const line_iterator& x, const line_iterator& y) const {
    return lexicographical_compare(x.first, x.second, y.first, y.second);
  }
};

struct strtab_print {
  ostream& out;
  strtab_print(ostream& os) : out(os) {}
  void operator()(const line_iterator& s) const {
    copy(s.first, s.second, ostream_iterator<char>(out));
  }
};

int main() {
  vector<char> strtab;
  char ch;
  while(cin.get(ch)) {
    strtab.push_back(ch);
  }

  vector<line_iterator> lines;
  auto cur = strtab.begin();
  while(cur != strtab.end()) {
    auto next = find(cur, strtab.end(), '\n');
    if (next != strtab.end()) {
      ++next;
    }
    lines.push_back(make_pair(cur, next));
    cur = next;
  }

  sort(lines.begin(), lines.end(), strtab_cmp());
  for_each(lines.begin(), lines.end(), strtab_print(cout));
  system("pause");
  return 0;
}