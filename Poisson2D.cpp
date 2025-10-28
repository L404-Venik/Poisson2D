#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <ctime>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

#include "CSRMatrix.h"
#include "Variant7.h"


std::vector<double> CustomRealization(const CSRMatrix& A, const std::vector<double>& F)
{
	int n = A.m_iRows;
	int M = std::sqrt(n);
	const int max_iter = n;
	const double delta = 0.1;

	std::vector<double> omega(n, 0.0);
	std::vector<double> r = F;          // r0 = F - A*x = F
	std::vector<double> p;
	std::vector<double> Ap(n, 0.0), z(n, 0.0);
	std::vector<double> D = A.GetDiagonal();
	std::vector<double> r_norms;

	for (int i = 0; i < n; ++i)
		z[i] = r[i] / D[i];

	p = z;
	double rz_old = DotProduct(z, r);

	for (int it = 0; it < max_iter; it++)
	{
		Ap = A.VectorMultiply(p);

		double pAp = DotProduct(p, Ap);

		double alpha = rz_old / pAp;

		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			omega[i] += alpha * p[i];

			r[i] -= alpha * Ap[i];

			z[i] = r[i] / D[i];
		}

		double rz_new = DotProduct(z, r);

		if (std::sqrt(rz_new) < delta)
		{
			std::cout << "converged in " << it << " stepts\n";
			break; // converged
		}
		//r_norms.push_back(rz_new);

		double beta = rz_new / rz_old;

		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			p[i] = z[i] + beta * p[i];
		}

		rz_old = rz_new;
	}

	/*std::ofstream ResultFile("residual.txt");
	PrintFlatMatrix(ResultFile, r_norms, r_norms.size(),1);
	ResultFile.close();*/

	return omega;
}


std::vector<double> StdRealization(const CSRMatrix& A, const std::vector<double>& F)
{
	int n = A.m_iRows;
	int M = std::sqrt(n);
	const int max_iter = n;
	const double delta = 0.1;

	std::vector<double> omega(n, 0.0);
	std::vector<double> r = F;          // r0 = F - A*x = F
	std::vector<double> p;
	std::vector<double> Ap(n, 0.0), z(n, 0.0);
	std::vector<double> D = A.GetDiagonal();
	std::vector<double> r_norms;

	for (int i = 0; i < n; ++i)
		z[i] = r[i] / D[i];

	p = z;
	double rz_old = DotProduct(z, r);

	for (int it = 0; it < max_iter; it++)
	{
		Ap = A.VectorMultiply(p);

		double pAp = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

		double alpha = rz_old / pAp;


		std::transform(p.begin(), p.end(), omega.begin(), omega.begin(),
			[alpha](double pi, double oi) { return oi + alpha * pi; });

		std::transform(Ap.begin(), Ap.end(), r.begin(), r.begin(),
			[alpha](double Ap, double r) { return r - alpha * Ap; });

		std::transform(r.begin(), r.end(), D.begin(), z.begin(),
			[](double r, double D) { return r / D; });

		double rz_new = std::inner_product(z.begin(), z.end(), r.begin(), 0.0);

		if (std::sqrt(rz_new) < delta)
		{
			std::cout << "converged in " << it << " stepts\n";
			break; // converged
		}
		//r_norms.push_back(rz_new);

		double beta = rz_new / rz_old;
		std::transform(z.begin(), z.end(), p.begin(), p.begin(),
			[beta](double zi, double pi) { return zi + beta * pi; });

		rz_old = rz_new;
	}

	/*std::ofstream ResultFile("residual.txt");
	PrintFlatMatrix(ResultFile, r_norms, r_norms.size(),1);
	ResultFile.close();*/

	return omega;
}


std::vector<double> ConjugateGradient(const CSRMatrix& A, const std::vector<double>& F)
{
	return CustomRealization(A, F);
	//return StdRealization(A, F);
}

size_t GetMilisecondsCount()
{
	using namespace std::chrono;
	return  duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


int main()
{
	int N, M, NumThreads; // X axis partitioned to M segments, Y - to N
	M = 400;
	N = 600;
	NumThreads = 1;
	//std::cin >> M >> N >> NumThreads;

	omp_set_num_threads(NumThreads);

	CSRMatrix A;
	std::vector<double> F, omega; // these are matrixes, just flatten
	CreateMatrixesV7(A, F, M, N);

	size_t avgTime = 0;
	int Passes = 3;
	for(int i = 0; i < Passes; i ++)
	{
		size_t start = GetMilisecondsCount();
		omega = ConjugateGradient(A, F);
		size_t end = GetMilisecondsCount();

		std::cout << i << " iteration - "<< end - start << " ms" << std::endl;

		avgTime += end - start;
	}
	avgTime /= Passes;


	std::cout << "average " << avgTime << " ms" << std::endl;

	/*std::string ResultFileName = "Result" + std::to_string(M) + "x" + std::to_string(N) + ".txt";
	std::ofstream ResultFile(ResultFileName);
	PrintFlatMatrix(ResultFile, omega, N + 1, M + 1);
	ResultFile.close();*/
}
