#include<iostream>
#include<cmath>
using namespace std;

// 多项式模板类，order 代表多项式的阶数， coefType代表多项式的系数数据类型
// 自变量永远都是一维的，在实数范围内取值(double类型)
template<int order,class coefType>
class polynomial
{
	enum {MAX_DEGREE = 15};
private:
	int degree;                // 多项式的阶
	coefType coef[order + 1];  // 多项式各项系数,比阶数多一个
	                           // 阶数由低到高排列
public:
	~polynomial(){ }
	polynomial();              // 默认构造 0 多项式  
	polynomial(const polynomial& p);
	polynomial(int n, coefType p[]);
	polynomial(coefType k);

	// 获取基本信息
	int get_degree() const { return degree; } // 返回多项式的次数
	coefType co(int n) const;         // 取 n 次项系数
	coefType value(double x) const;   // 取 x 点处的值
	coefType d_value(double x) const; // 取 x 点的导数值

	polynomial derivation() const;    // 求导数

	// 运算符重载
	polynomial operator+(const polynomial& p) const;
	polynomial operator-(const polynomial& p) const;
	polynomial operator-() const;
	polynomial operator*(const polynomial& p) const;
	polynomial operator*(double k) const;
	polynomial& operator=(const polynomial& p);
	polynomial& operator=(coefType k);

	// 友元
	friend polynomial& operator*(double k, const polynomial& p)
	{
		coefType ans[MAX_DEGREE];
		for (int  i = 0; i <= p.degree; i++)
		{
			ans[i] = p.coef[i] * k;
		}
		return polynomial(p.degree, ans);
	}

	// ostream输出
	friend ostream& operator<<(ostream& os, const polynomial<order, coefType>& p);
};

// 默认构造函数，构造0多项式
template<int order, class coefType>
polynomial<order, coefType>::polynomial()
{
	degree = 0;
	for (int i = 0; i < degree + 1; i++)
	{
		coef[i] = 0;  
	}
}

// 构造函数
template<int order,class coefType>
polynomial<order, coefType>::polynomial(int n, coefType p[])
{
	degree = n;
	for (int i = 0; i < degree + 1; i++)
	{
		coef[i] = p[i];
	}
}

// 复制构造函数
template<int order,class coefType>
polynomial<order, coefType>::polynomial(const polynomial& p)
{
	degree = p.degree;
	for (int i = 0; i < degree; i++)
	{
		coef[i] = p.coef[i];
	}
}

// 构造函数：常数多项式
template<int order,class coefType>
polynomial<order, coefType>::polynomial(coefType k)
{
	degree = 0;
	coef[0] = k;
}

// 取多项式的 n 阶项系数
template<int order,class coefType>
coefType polynomial<order, coefType>::co(int n) const
{
	return coef[n];
}

// 取多项式在点 x 处的值
template<int order, class coefType>
coefType polynomial<order, coefType>::value(double x) const
{
	coefType ans;
	for (int i = 0; i < degree+1; i++)
	{
		ans += coef[i] * pow(x, i);
	}
	return ans;
}

// 求多项式的导数多项式
template<int order, class coefType>
polynomial<order,coefType> polynomial<order, coefType>::derivation() const
{
	if (degree == 0)
	{
		return polynomial<order, coefType>();
	}
	coefType ans[MAX_DEGREE];
	for (int i = 0; i < degree; i++)
	{
		ans[i] = coef[i + 1] * (double)(i + 1);
	}
	return polynomial<order, coefType>(degree - 1, ans);
}

// 求多项式的导数在点 x 处的值
template<int order,class coefType>
coefType polynomial<order, coefType>::d_value(double x) const
{
	polynomial<order, coefType> q = this->derivation();
	return q.value(x);
}

// 重载：多项式相加
template<int order,class coefType>
polynomial<order,coefType> 
polynomial<order, coefType>::operator+(const polynomial& p) const
{
	int high = (degree > p.degree) ? degree : p.degree; // 次数高的那个的次数
	int low = (degree < p.degree) ? degree : p.degree;  // 次数低的那个的次数
	coefType ans[MAX_DEGREE];
	for (int i = 0; i <= high; i++)
	{
		if (i <= low)
			ans[i] = coef[i] + p.coef[i];
		else
			ans[i] = (degree > p.degree) ? coef[i] : p.coef[i]; // 次数高的那个的系数
	}
	return polynomial<order,coefType>(high,ans);
}

// 重载：多项式相减
template<int order, class coefType>
polynomial<order, coefType>
polynomial<order, coefType>::operator-(const polynomial& p) const
{
	int high = (degree > p.degree) ? degree : p.degree; // 次数高的那个的次数
	int low = (degree < p.degree) ? degree : p.degree;  // 次数低的那个的次数
	coefType ans[MAX_DEGREE];
	for (int i = 0; i <= high; i++)
	{
		if (i <= low)  // 两个多项式公共项
			ans[i] = coef[i] - p.coef[i];
		else         // 次数高的多项式独有的项
			ans[i] = (degree > p.degree) ? coef[i] : -p.coef[i]; // 次数高的那个的系数
	}
	return polynomial<order, coefType>(high, ans);
}

// 重载：多项式取负号
template<int order,class coefType>
polynomial<order,coefType>
polynomial<order, coefType>::operator-() const
{
	coefType ans[MAX_DEGREE];
	for (int i = 0; i < degree; i++)
	{
		ans[i] = -coef[i];
	}
	return polynomial<order, coefType>(degree, ans);
}

// 重载：多项式相乘（系数类型都是vec时相乘无意义）
template<int order, class coefType>
polynomial<order, coefType>
polynomial<order, coefType>::operator*(const polynomial& p) const
{
	int m = degree + p.degree; // 两个多项式次数之和
	if (m == 0)
		return polynomial<order, coefType>(coef[0] * p.coef[0]);
	coefType ans[MAX_DEGREE];
	for (int i = 0; i <= m ; i++)
	{
		for (int j = 0 ; j <= i; j++)
			ans[j] += coef[j] * p.coef[i - j];
	}
	return polynomial<order, coefType>(m, ans);
}

// 重载：多项式乘一个实数
template<int order, class coefType>
polynomial<order, coefType>
polynomial<order, coefType>::operator*(double k) const
{
	coefType ans[MAX_DEGREE];
	for (int i = 0; i <= degree; i++)
	{
		ans[i] = coef[i] * k;
	}
	return polynomial<order, coefType>(degree, ans);
}

// 重载：把一个多项式赋值给另一个多项式
template<int order, class coefType>
polynomial<order, coefType>&
polynomial<order, coefType>::operator=(const polynomial& p)
{
	if (this == &p)
		return *this;
	degree = p.degree;
	coef = new coefType(degree + 1);
	for (int i = 0; i <= degree; i++)
	{
		coef[i] = p.coef[i];
	}
	return *this;
}

// 重载：把一个“数”赋值给一个多项式
template<int order, class coefType>
polynomial<order, coefType>&
polynomial<order, coefType>::operator=(coefType k)
{
	for (int i = 0; i <= degree; i++)
	{
		coef[i] = 0;
	}
	degree = 0;
	coef[0] = k;
	return *this;
}

// 友元：ostream输出
template<int order, class coefType>
ostream& operator<<(ostream& os, const polynomial<order, coefType>& p)
{

}

int main () {
	return 0;
}

