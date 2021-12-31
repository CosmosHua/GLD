// check_match.cpp : 此文件包含main()函数。程序执行将在此处开始并结束。
// 给出两幅图像及相应的同名像点文件，对同名像点进行评估，并给出有问题的同名点
//
// 命令行参数： matchfile toler [resfile]
//	matchfile 为已获取的同名像点文件，第一行为总的点数，以后每行为图像同名像点坐标：x0 y0 x1 y1
//  toler 为判断容差，单位为pix，大于此值的点在resfile中被标记为错误点
//	resfile 结果文件名，按json格式存贮
//  命令行示例：check_match F:/BaiduNetdiskWorkspace/python/0389_0295.txt 5 result.json

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#define TRUE  1
#define FALSE 0
#define NUM 50
//#define MAXFLOAT 1e35
#define MaxOrignalErrorFunctionThreshold 500

#define SwapDouble(d0, d1) { double tmp = d0; d0 = d1; d1 = tmp; }
// 点为列向量，矩阵按行存贮
class CPoint2D
{
public:
	double x, y;
};

class CGridPoint
{
public:
	CPoint2D p0, p1;
};

class CPoint3D
{
public:
	double m_pt[3];
};
class CMatrix3D
{
public:
	double m_mat[3][3];
public:
	void setZero() { memset(m_mat, 0, 9 * sizeof(double)); };
	void setUnit() { setZero();  m_mat[0][0] = m_mat[1][1] = m_mat[2][2] = 1; };
	void setRotateX(double ang)
	{
		double sina, cosa;
		sina = sin(ang);
		cosa = cos(ang);
		setUnit();
		m_mat[1][1] = cosa;
		m_mat[1][2] = -sina;
		m_mat[2][1] = sina;
		m_mat[2][2] = cosa;
	};
	void setRotateY(double ang)
	{
		double sina, cosa;
		sina = sin(ang);
		cosa = cos(ang);
		setUnit();
		m_mat[0][0] = cosa;
		m_mat[0][2] = sina;
		m_mat[2][0] = -sina;
		m_mat[2][2] = cosa;
	};
	void setRotateZ(double ang)
	{
		double sina, cosa;
		sina = sin(ang);
		cosa = cos(ang);
		setUnit();
		m_mat[0][0] = cosa;
		m_mat[0][1] = -sina;
		m_mat[1][0] = sina;
		m_mat[1][1] = cosa;
	};
	CMatrix3D timesMatrix(CMatrix3D mat)
	{
		CMatrix3D mm;
		int i, j, k;
		for (j = 0; j < 3; j++)
		{
			for (i = 0; i < 3; i++)
			{
				mm.m_mat[j][i] = 0;
				for (k = 0; k < 3; k++)
				{
					mm.m_mat[j][i] += mat.m_mat[j][k] * m_mat[k][i];
				}
			}
		}
		return mm;
	};
	CPoint3D transPoint(CPoint3D pt0)
	{
		int j, k;
		CPoint3D pt1;
		for (j = 0; j < 3; j++)
		{
			pt1.m_pt[j] = 0;
			for (k = 0; k < 3; k++)
			{
				pt1.m_pt[j] += m_mat[j][k] * pt0.m_pt[k];
			}
		}
		return pt1;

	};
};

class CMatchList
{
protected:
	CMatrix3D m_mat0, m_mat1;
	CPoint3D *m_p0list, *m_p1list;
	char *m_deltag;
	int m_pnum;
	int m_width0, m_height0, m_width1, m_height1;
	int m_ImageSize; // 按两图的最大计
	double m_fx0, m_fy0, m_fx1, m_fy1;

public:
	bool CalInitPosture(const char *matchfile);
	bool GetEpipolarForFrameTypeGragh(double res[16], int count[NUM]);
	int outputError(double res[16], int count[NUM], double toler, const char *outfile);
};

int LinarEquation(int dim, double *a, double *res, double toler)
{	// 解线性方程组,解完后系数矩阵已破坏
	int i, j, k, D, tag;
	double vmax;

	tag = TRUE;
	D = dim + 1;
	for (i = 0; i < dim; i++)
	{
		vmax = fabs(a[i*D + i]);
		k = i;
		for (j = i + 1; j < dim; j++)
		{
			//找最大元素
			if (fabs(a[j*D + i]) > vmax)
			{
				vmax = fabs(a[j*D + i]);
				k = j;
			}
		}
		if (fabs(vmax) < toler)
		{
			tag = FALSE; //  方程无唯一解
			continue;	// 若为无穷多解,则求其一解
		}
		// 两行互换
		if (i != k)
		{
			for (j = i; j < D; j++)
			{
				SwapDouble(a[i*D + j], a[k*D + j]);
			}
		}

		// 对角线元素变为1
		vmax = a[i*D + i];
		for (j = i; j < D; j++)
		{
			a[i*D + j] /= vmax;
		}

		// 对角线以下元素变为0
		for (j = i + 1; j < dim; j++)
		{
			for (k = i + 1; k < D; k++)
			{
				a[j*D + k] -= a[i*D + k] * a[j*D + i];
			}
			a[j*D + i] = 0;
		}
	}

	// 至此应已变为三角阵,反向化简
	for (i = dim - 1; i >= 0; i--)
	{
		if (a[i*D + i] < toler)
		{
			if (a[i*D + dim] > sqrt(toler))
				return FALSE;
			a[i*D + i] = 1;
			a[i*D + dim] = 1; // 该变量可取任意值
		}
		for (j = i - 1; j >= 0; j--)
		{
			a[j*D + dim] -= a[j*D + i] * a[i*D + dim];
			a[j*D + i] = 0;
		}
	}
	for (i = 0; i < dim; i++)
		res[i] = a[i*D + dim];
	return tag;
}

int LinarEquationX(double *mat, int dim, int num, double toler, double *res)
{	// 用最小二乘法解矛盾的线性方程组,解完后系数矩阵未破坏
	// 也可以用此函数解一般的线性方程组,以便不破坏系数矩阵
	// mat为系数矩阵,dim为变量个数,num为方程个数
	// 系数矩阵按行为序,即先是第一个方程的系数,然后是第二个

	int i, j, k, D, tag;
	double *a;

	if (num < dim)
		return FALSE;

	//求相应的最小二乘线性方程组系数矩阵
	D = dim + 1;
	a = new double[D*dim];
	if (num == dim)
		memcpy(a, mat, dim*D * sizeof(double));
	else
	{
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < D; j++)
			{
				a[i*D + j] = 0;
				for (k = 0; k < num; k++)
				{
					a[i*D + j] += mat[k*D + i] * mat[k*D + j];
				}
			}
		}
	}
	tag = LinarEquation(dim, a, res, toler);
	delete[] a;
	return tag;
}

bool CMatchList::CalInitPosture(const char *matchfile)
{
	int EfficientNum = 0;
	int i = 0, j = 0, k = 0;
	double ang0, ang1, S1;	// 偏移量与旋转角
	int  tag;
	int xnum;
	double *mm;
	double x0, y0, x1, y1, res[3];

	const float precision = 2.1f;  //1.732*Pre~=,Pre~=0.9
	int ReturnFlag = 1;
	double yaverage = 0;
	CMatrix3D m0, m1;
	FILE *fp;

	//tag = fopen_s(&fp, matchfile, "rt");
	fp = fopen(matchfile, "rt");
	if (fp == NULL)
		return 0;

	fscanf(fp, "%d", &m_pnum);
	fscanf(fp, "%lf %lf %lf %lf", &m_fx0, &m_fy0, &m_fx1, &m_fy1); // 两个相机已统一到同一个焦距1,这里只是为了计算像素误差
	fscanf(fp, "%d %d %d %d", &m_width0, &m_height0, &m_width1, &m_height1);
	m_ImageSize = 0;
	if (m_width0 > m_ImageSize)
		m_ImageSize = m_width0;
	if (m_height0 > m_ImageSize)
		m_ImageSize = m_height0;
	if (m_width1 > m_ImageSize)
		m_ImageSize = m_width1;
	if (m_height1 > m_ImageSize)
		m_ImageSize = m_height1;

	m_p0list = new CPoint3D[m_pnum];
	m_p1list = new CPoint3D[m_pnum];
	m_mat0.setUnit();
	m_mat1.setUnit();

	for (i = 0; i < m_pnum; i++)
	{
		fscanf(fp, "%lf %lf %lf %lf", &x0, &y0, &x1, &y1);
		// 要求为Z=1平面上的坐标系。
		m_p0list[i].m_pt[0] = x0;
		m_p0list[i].m_pt[1] = y0;
		m_p0list[i].m_pt[2] = 1;
		m_p1list[i].m_pt[0] = x1;
		m_p1list[i].m_pt[1] = y1;
		m_p1list[i].m_pt[2] = 1;
	}
	fclose(fp);

	mm = new double[m_pnum * 4];
	m_deltag = new char[m_pnum];
	memset(m_deltag, 0, m_pnum * sizeof(char));
	//最小二乘解算初值
	k = 0;
	for (i = 0; i < m_pnum; i++)
	{
		if (m_deltag[i])
			continue;
		x0 = m_p0list[i].m_pt[0];
		y0 = m_p0list[i].m_pt[1];
		x1 = m_p1list[i].m_pt[0];
		y1 = m_p1list[i].m_pt[1];

		// 算法思想：两图分别旋转，A图上下平移，B图比例，可致两图的y一致
		//delta = x0*sina + y0*cosa - Dy - (x1*sinb + y1*cosb)*S1;
		// 此解算只有当a很小时才成立！！！
		mm[k++] = x0;		//sina/cosa=res[0]
		mm[k++] = -x1;		//sinb*S1/cosa=res[2]
		mm[k++] = -y1;		//cosb*S1/cosa=res[3]
		mm[k++] = -y0;
	}
	xnum = k / 4;
	tag = LinarEquationX(mm, 3, xnum, 1e-7, res);
	if (tag == 0)
	{
		xnum = 0;
		ReturnFlag = 0;
		delete[] mm;
		return FALSE;
	}

	double c0, x, y, dr1, dr2;
	ang0 = atan(res[0]);
	c0 = cos(ang0);
	y = cos(ang0)*res[1];
	x = cos(ang0) * res[2];
	//Dy = res[1] * cos(ang0);
	ang1 = atan2(y, x);
	S1 = res[2] * cos(ang0) / cos(ang1);
	dr1 = sin(ang1) / cos(ang0) - res[1];
	dr2 = cos(ang1) / cos(ang0) - res[2];

	m0.setRotateZ(ang0);
	m1.setRotateZ(ang1);
	m_mat0 = m_mat0.timesMatrix(m0);
	m_mat1 = m_mat1.timesMatrix(m1);

	delete[] mm;
	return 1;
}

/////////////////////////  框幅式相机核线模型    ///////////////////////////////////
bool CMatchList::GetEpipolarForFrameTypeGragh(double res[16], int count[NUM])
//自己设计的核线生成算法，求取核线参数res，参数意义见内部函数
{
	if (m_pnum <= 3)
		return NULL;

	int EfficientNum = 0;
	int i = 0, j = 0, k = 0;

	double dy;
	int  tag;
	int xnum;
	double *mm, err, lasterr, maxd;
	double x0, y0, x1, y1, sill;

	const float precision = 2.1f;  //1.732*Pre~=,Pre~=0.9
	int ReturnFlag = 1;
	double yaverage = 0;

	CMatrix3D m0, m1;
	CPoint3D p3d;

	mm = new double[m_pnum * 6];
	err = 1000;
	int maxstep, ready;
	maxstep = 20;
	ready = 0;
	sill = MaxOrignalErrorFunctionThreshold;
	lasterr = MAXFLOAT;
	while (maxstep)
	{
		maxstep--;
		k = 0;
		for (i = 0; i < m_pnum; i++)
		{
			if (m_deltag[i])
				continue;
			p3d = m_mat0.transPoint(m_p0list[i]);
			x0 = p3d.m_pt[0] / p3d.m_pt[2];
			y0 = p3d.m_pt[1] / p3d.m_pt[2];

			p3d = m_mat1.transPoint(m_p1list[i]);
			x1 = p3d.m_pt[0] / p3d.m_pt[2];
			y1 = p3d.m_pt[1] / p3d.m_pt[2];

			mm[k++] = x0;
			mm[k++] = y0 * x0;
			mm[k++] = -x1;
			mm[k++] = -y1 * x1;
			mm[k++] = 1;
			mm[k++] = y1 - y0;
		}
		xnum = k / 6;
		tag = LinarEquationX(mm, 5, xnum, 1e-7, res);
		if (tag == 0)
		{
			xnum = 0;
			ReturnFlag = 0;
			return FALSE;
		}

		m0.setRotateZ(res[0]);
		m_mat0 = m_mat0.timesMatrix(m0);
		m0.setRotateY(res[1]);
		m_mat0 = m_mat0.timesMatrix(m0);

		m1.setRotateZ(res[2]);
		m_mat1 = m_mat1.timesMatrix(m1);
		m1.setRotateY(res[3]);
		m_mat1 = m_mat1.timesMatrix(m1);
		m1.setRotateX(res[4]);
		m_mat1 = m_mat1.timesMatrix(m1);

		k = 0;
		err = 0;
		maxd = 0;
		memset(count, 0, NUM * sizeof(int));
		for (i = 0; i < m_pnum; i++)
		{
			p3d = m_mat0.transPoint(m_p0list[i]);
			x0 = p3d.m_pt[0] / p3d.m_pt[2];
			y0 = p3d.m_pt[1] / p3d.m_pt[2];

			p3d = m_mat1.transPoint(m_p1list[i]);
			x1 = p3d.m_pt[0] / p3d.m_pt[2];
			y1 = p3d.m_pt[1] / p3d.m_pt[2];

			// 求偏移距离直方图
			dy = fabs(y1 - y0)*m_fx0*m_ImageSize;
			j = (int)dy;
			if (j >= NUM)
				j = NUM - 1;
			count[j]++;
			if (m_deltag[i])
				continue;
			if (ready)
			{
				if (dy >= sill)
				{
					m_deltag[i] = 1;
					continue;
				}
				if (dy > maxd)
					maxd = dy;
			}
			err += dy;
			k++;
		}
		err = err / k;
		if (!ready)
		{
			double tt = 0;
			for (int ki = 0; ki < 5; ki++)
			{
				tt += res[ki] * res[ki];
			}
			if (tt < 1e-10)
				ready = 1;
		}
		else
		{
			// 确定下一步的sill值
			sill *= 0.9;
			if (sill > maxd*0.9)
				sill = maxd * 0.9;
			if (fabs(lasterr - err) < 0.001)
				break;
			lasterr = err;
		}
	}

	delete[] mm;
	mm = NULL;
	delete[] m_deltag;
	m_deltag = NULL;

	// 返回结果参数
	res[0] = sill;
	res[1] = err;

	return TRUE;
}

int CMatchList::outputError(double res[16], int count[NUM], double toler, const char *outfile)
{
	double err;
	double x0, y0, x1, y1, dy, sill;
	int i, j, tag;
	FILE *fp = NULL;

	//tag = fopen_s(&fp, outfile, "wt");
	fp = fopen(outfile, "wt");
	if (fp == NULL) return 0;

	err = res[1];
	sill = res[0];
	fprintf(fp, "{\n");
	fprintf(fp, "    \"sill\": %.2lf,\n", sill);
	fprintf(fp, "    \"toler\": %.3lf,\n", toler);
	fprintf(fp, "    \"err\": %.3lf,\n", err);
	fprintf(fp, "    \"Matrix0\": \"");
	for (j = 0; j < 3; j++)
		for (i=0; i<3; i++)
			fprintf(fp, " %.7lf", m_mat0.m_mat[j][i]);
	fprintf(fp, "\",\n");
	fprintf(fp, "    \"Matrix1\": \"");
	for (j = 0; j < 3; j++)
		for (i = 0; i < 3; i++)
			fprintf(fp, " %.7lf", m_mat1.m_mat[j][i]);
	fprintf(fp, "\",\n");

	int i0 = 0;
	fprintf(fp, "    \"count\": \"");
	while (i0 < NUM)
	{
		fprintf(fp, " %d", count[i0]);
		i0++;
	}
	fprintf(fp, "\",\n");

	int ltag = 0;
	CPoint3D p3d;
	fprintf(fp, "    \"errMatch\": {\n");
	for (i = 0; i < m_pnum; i++)
	{
		p3d = m_mat0.transPoint(m_p0list[i]);
		x0 = p3d.m_pt[0] / p3d.m_pt[2];
		y0 = p3d.m_pt[1] / p3d.m_pt[2];

		p3d = m_mat1.transPoint(m_p1list[i]);
		x1 = p3d.m_pt[0] / p3d.m_pt[2];
		y1 = p3d.m_pt[1] / p3d.m_pt[2];

		// 因为是图像规格化坐标系，转化到象素坐标与图像大小有关
		dy = (y1 - y0)*m_fx0*m_ImageSize;
		// 求偏移距离直方图
		if (fabs(dy) <= toler) continue;
		if (ltag) fprintf(fp, ",\n");
		fprintf(fp, "        \"P%d\": %.2lf", i, dy);
		ltag++;
	}
	fprintf(fp, "\n    }\n}\n");
	fclose(fp); return 1;
}

int main(int argc, char *argv[])
{
	if (argc<3) return 0;
	const char *matchfile = argv[1];
	double toler = atof(argv[2]);

	std::string x = argv[1];
	x.replace(x.rfind("."), 5, ".json");
	const char *resfile = x.c_str();
	if (argc>3) resfile = argv[3];

	int count[NUM] = {0}; //initial
	CMatchList mt; double res[16];
	int tag = mt.CalInitPosture(matchfile);
	tag = mt.GetEpipolarForFrameTypeGragh(res, count);
	mt.outputError(res, count, toler, resfile);
	return tag;
}//end main

