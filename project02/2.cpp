#include <iostream>

using namespace std;

typedef long long ll;
const ll maxn = 1e3 + 7;
ll Map[maxn][maxn]; //存图
ll P[maxn][maxn];
ll N, M;
void Path(ll u, ll v)
{
  if (P[u][v] == 0)
    return;
  Path(u, P[u][v]);
  cout << P[u][v] << ' ';
  Path(P[u][v], v);
}

int main()
{
  //	freopen(".../.txt","w",stdout);
  cin >> N >> M;
  ll i, j, k;
  memset(Map, 0x3f, sizeof(Map)); //一开始赋一个很大的值
  memset(P, 0, sizeof(P));        //初始化
  for (i = 1; i <= N; i++)
    Map[i][i] = 0; //自身为0
  for (i = 1; i <= M; i++)
  {
    ll u, v, va;
    cin >> u >> v >> va;
    Map[u][v] = va;
  }
  for (k = 1; k <= N; k++)
  {
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        if (Map[i][k] + Map[k][j] < Map[i][j])
        {
          Map[i][j] = Map[i][k] + Map[k][j]; //更新最短路径值
          P[i][j] = k;                       //记录中间点
        }
      }
    }
  }
  for (i = 1; i <= N; i++)
  {
    for (j = 1; j <= N; j++)
    {
      cout << i << "-->" << j << "  ";
      cout << Map[i][j] << ' ';
      if (Map[i][j] == 0)
      {
        cout << endl;
        continue;
      }
      cout << i << ' ';
      Path(i, j);
      cout << j << endl;
    }
  }
  return 0;
}
