#include <conio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <cstring>
#include <iostream>
#include <string>
using namespace std;

namespace SSR {
const double pi = acos(-1);
const double eps = 0.01;
long ground_light = 1;
struct vec {
  double eps_ = 0.01;
  double x, y, z;
  vec(double a = 0, double b = 0, double c = 0) { x = a, y = b, z = c; };
  double Length() { return sqrt(x * x + y * y + z * z); }
  vec Unit() {
    double l = sqrt(x * x + y * y + z * z);
    return vec(x, y, z) / l;
  }
  vec operator==(const vec &A) {
    return (abs(x - A.x) <= eps_ && abs(y - A.y) <= eps_);
  }
  double operator*(const vec &A) { return x * A.x + y * A.y + z * A.z; }
  vec operator*(const double k) {
    vec res;
    res.x = this->x * k;
    res.y = this->y * k;
    res.z = this->z * k;
    return res;
  }
  vec operator/(const double k) {
    vec res;
    res.x = this->x / k;
    res.y = this->y / k;
    res.z = this->z / k;
    return res;
  }
  vec operator+(const vec &A) {
    vec res;
    res.x = this->x + A.x;
    res.y = this->y + A.y;
    res.z = this->z + A.z;
    return res;
  }
  vec operator-(const vec &A) {
    vec res;
    res.x = this->x - A.x;
    res.y = this->y - A.y;
    res.z = this->z - A.z;
    return res;
  }

  vec crs(vec A, vec B) {
    vec res;
    res.x = A.y * B.z - A.z * B.y;
    res.y = -A.x * B.z + A.z * B.x;
    res.z = A.x * B.y - A.y * B.x;
    return res;
  }
};
struct Camera {
  double ang = pi / 3;  // 半视角
  double f = 10;        // 焦距
  double m = f * tan(ang);

  double sight = 500;  // 视距
  double step = 0.1;
  double ang_step = pi / 45;

  vec pos = vec(0, 0, 0);

  double theta = 0, phi = pi / 2;
  // phi=[0,pi],theta=[0,2pi]

  vec direct = vec(1, 0, 0);
  vec e_x;
  vec e_y;

  void set() {
    if (theta >= 2 * pi) theta -= 2 * pi;
    if (theta < 0) theta += 2 * pi;
    m = f * tan(ang);
    ang_step = (pi / 45) * (ang / (pi / 3));
    direct = vec(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    e_x = vec(direct.y, -direct.x, 0);
    e_y = vec(-direct.x * direct.z, -direct.y * direct.z,
              1 - direct.z * direct.z);
    e_x = e_x.Unit();
    e_y = e_y.Unit();
  }
};
struct Screen {
  char pix[110][110];
  const char color[9] = {'@', '%', '#', '*', '+', '=', '-', '.', ' '};
  //                      100, 90,  80,  70   50   30   10   5   0
  void print(Camera cam) {
    HANDLE hOutput;
    COORD coord = {0, 0};
    hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cci;
    GetConsoleCursorInfo(hOutput, &cci);
    cci.bVisible = false;
    SetConsoleCursorInfo(hOutput, &cci);
    string A = "\n";
    for (int j = 75; j >= 25; j--) {
      for (int i = 10; i <= 95; i++) {
        A += pix[i][j];
        A += ' ';
      }
      A += '\n';
    }
    SetConsoleCursorPosition(hOutput, coord);
    printf("\n%s", A.c_str());
    printf("now pos:%.2f %.2f %.2f ang:%.2f\n", cam.pos.x, cam.pos.y, cam.pos.z,
           cam.ang * 180 / pi);
  }
  void Color(double light, int i, int j) {
    if (light >= 330)
      pix[i][j] = color[0];
    else if (light >= 280)
      pix[i][j] = color[1];
    else if (light >= 230)
      pix[i][j] = color[2];
    else if (light >= 180)
      pix[i][j] = color[3];
    else if (light >= 130)
      pix[i][j] = color[4];
    else if (light >= 70)
      pix[i][j] = color[5];
    else if (light >= 30)
      pix[i][j] = color[6];
    else if (light >= 10)
      pix[i][j] = color[7];
    else
      pix[i][j] = color[8];
    return;
  }
};
struct Ball {
  vec pos = vec(0, 0, 0);
  double r = 0;
  double light = 100;
};
// ax^2+bx+c=0
double delta(double a, double b, double c) { return b * b - 4 * a * c; }
double min(double a, double b) { return a > b ? b : a; }
void Get_pic(Camera cam, Screen &S, Ball *ball, int ball_num) {
  //[i,j]
  for (int i = -50; i <= 50; i++) {
    for (int j = -50; j <= 50; j++) {
      vec n = cam.direct * cam.f + cam.e_x * (i * 0.01 * cam.m) +
              cam.e_y * (j * 0.01 * cam.m);
      vec p = cam.pos;
      n = n.Unit();
      //
      double light = 0;
      int now_k = -1;
      double tot_lamda = 0;
    //
    again_:
      double lamda = -1;
      // ball
      for (int k = 0; k < ball_num; k++) {
        vec o = ball[k].pos;
        double r = ball[k].r;
        double del = delta(1, (n * (p - o)) * 2, (p - o) * (p - o) - r * r);
        if (k == now_k || del < 0) continue;
        if ((n * (o - p) - sqrt(del) / 2 > 0) &&
            (lamda < 0 || lamda > n * (o - p) - sqrt(del) / 2)) {
          now_k = k, lamda = n * (o - p) - sqrt(del) / 2;
        }
      }
      // ground
      if (n.z < 0) {
        double t = abs(p.z / n.z);
        vec o = p + n * t;
        if (lamda < 0 || lamda > t) {
          tot_lamda += t;
          if ((((int)o.x / 1) % 2) ^ (((int)o.y / 1) % 2))
            light += ground_light / cbrt(tot_lamda * tot_lamda);
          S.Color(light, i + 50, j + 50);
          continue;
        }
      }

      // color
      if (lamda < 0 || tot_lamda > cam.sight)
        S.Color(light, i + 50, j + 50);
      else {
        tot_lamda += lamda;
        light += ball[now_k].light / (sqrt(tot_lamda));
        double temp = (p + (n * lamda) - ball[now_k].pos) *
                      (p - ball[now_k].pos) / (ball[now_k].r * ball[now_k].r);
        vec tem = n;
        n = (p + (n * lamda) - ball[now_k].pos) * (2 * temp - 1) -
            (p - ball[now_k].pos);
        n = n.Unit();
        p = p + tem * lamda;
        goto again_;
      }
    }
  }
}

void Move(Camera &cam) {
  char key;
  cam.set();
  //
  if (kbhit()) {
    fflush(stdin);
    key = getch();
    vec direct;
    switch (key) {
      case 'w':
        direct = cam.direct;
        direct.z = 0;
        direct = direct.Unit();
        cam.pos = cam.pos + direct * cam.step;
        break;
      case 's':
        direct = cam.direct;
        direct.z = 0;
        direct = direct.Unit();
        cam.pos = cam.pos - direct * cam.step;
        break;
      case 'a':
        direct = cam.e_x;
        direct.z = 0;
        direct = direct.Unit();
        cam.pos = cam.pos - direct * cam.step;
        break;
      case 'd':
        direct = cam.e_x;
        direct.z = 0;
        direct = direct.Unit();
        cam.pos = cam.pos + direct * cam.step;
        break;
      case 'q':
        cam.pos = cam.pos + vec(0, 0, 1) * cam.step;
        break;
      case 'e':
        cam.pos = cam.pos - vec(0, 0, 1) * cam.step;
        break;
      // up
      case 72:
        if (cam.phi > cam.ang_step) cam.phi -= cam.ang_step;
        break;
      // down
      case 80:
        if (cam.phi <= pi - cam.ang_step) cam.phi += cam.ang_step;
        break;
      // right
      case 77:
        cam.theta -= cam.ang_step;
        break;
      // left
      case 75:
        cam.theta += cam.ang_step;
        break;
      // 视角
      case ',':
        if (cam.ang < pi / 2 - eps) cam.ang += cam.ang_step * 0.1;
        break;
      case '.':
        if (cam.ang > 0) cam.ang -= cam.ang_step * 0.1;
        break;
      default:
        break;
    }
  }
}

void main() {
  Screen S;
  Camera cam;
  cam.pos = vec(0, 0, 1);
  int ball_num = 7;
  Ball ball[ball_num];
  ball[0].pos = vec(0, -8, 4);
  ball[0].r = 5;
  ball[0].light = 300;
  ball[5].pos = vec(-100, 0, 50);
  ball[5].r = 50;
  ball[5].light = 100;
  ball[4].pos = vec(0, 0, 10);
  ball[4].r = 2;
  ball[4].light = 1000;
  ball[3].pos = vec(0, 2, 4);
  ball[3].r = 5;
  ball[3].light = 100;
  ball[1].pos = vec(8, 0, 0);
  ball[1].r = 1;
  ball[1].light = 2000;
  ball[2].pos = vec(8, 0, 0);
  ball[2].r = 1.5;
  ball[2].light = 300;
  ball[6].pos = vec(8, 0, 0);
  ball[6].r = 1.5;
  ball[6].light = 300;
  double time = 0;
  while (1) {
    Move(cam);
    ball[1].pos = vec(8 * cos(time * 0.1), 8 * sin(time * 0.1), 0.5);
    ball[1].light = 300 + 100 * sin(time * 2);
    ball[2].pos = vec(8 * cos(time * 1), 7 * sin(time * 2), 3);
    ball[6].pos = vec(8, 7, 3);
    ball[6].light = 400 + 200 * sin(time * 2);
    ball[4].pos = vec(8 * cos(time * 2), 7 * sin(time * 2), 10);
    ball[5].pos = vec(100 * cos(time * 0.01), 70 * sin(time * 0.05),
                      50 + 50 * cos(time * 0.05) * sin(time * 0.05));
    ball[0].pos = vec(0, -8 + 0.1 * cos(time * pi), 4 + 3 * sin(time * 0.1));
    Get_pic(cam, S, ball, ball_num);
    S.print(cam);
    // cout << "\033c";
    time += 0.005;
    ground_light = 1000 + 500 * sin(time * 2);
    Sleep(1);
  }
  return;
}
}  // namespace SSR
int main() {
  SSR::main();
  return 0;
}