#include <iostream>
#include <fstream>
#include <limits>
#include <ctime>

#include "CSRMatrix.h"
#include "Variant7.h"


std::vector<double> ConjugateGradient(const CSRMatrix& A, const std::vector<double>& F)
{
	int n = A.m_iRows;
	int M = std::sqrt(n);
	const int max_iter = n;
	const double delta = 0.1;

	std::vector<double> omega(n, 0.0);
	std::vector<double> r = F;           // r0 = F - A*x = F
	std::vector<double> p;
	std::vector<double> Ap(n, 0.0), z(n, 0.0);
	std::vector<double> D = A.GetDiagonal();

	for (int i = 0; i < n; ++i)
		z[i] = r[i] / D[i];

	p = z;
	double rz_old = DotProduct(z,r);

	for (int it = 0; it < max_iter; it++)
	{
		Ap = A.VectorMultiply(p);

		double pAp = DotProduct(p, Ap);

		double alpha = rz_old / pAp;

		double C_norm = 0.0;
		for (int i = 0; i < n; i++)
		{
			omega[i] += alpha * p[i];

			r[i] -= alpha * Ap[i];

			z[i] = r[i] / D[i];

			C_norm = std::max(C_norm, std::abs(r[i]));
		}

		double rz_new = DotProduct(z,r);
		//if(rz_new > rz_old)
		if (std::sqrt(rz_new) < delta) 
		//if (C_norm < delta)
		{
			break; // converged
		}

		double beta = rz_new / rz_old;
		for (int i = 0; i < n; i++)
		{
			p[i] = z[i] + beta * p[i];
		}

		rz_old = rz_new;
	}

	return omega;
}

int main()
{
	__int64 N, M; // X axis partitioned to M segments, Y - to N
	M = 12;
	N = 12;

	CSRMatrix A;
	std::vector<double> F, omega; // these are matrixes, just flatten
	CreateMatrixesV7(A, F, M, N);

	omega = ConjugateGradient(A, F);

	std::ofstream ResultFile("out.txt");
	//A.print(ResultFile);
	for (int i = 0; i <= N; i++)
	{
		for (int j = 0; j <= M; j++)
		{
			ResultFile << omega[i * (M + 1) + j] << ' ';
		}
		ResultFile << '\n';
	}
	ResultFile.close();
}
