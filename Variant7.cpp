#include "Variant7.h"
#include <numeric>
#include <limits>
#include <float.h>
#include <cmath>
#include <cassert>
#include <string>

inline bool IsCenter(double X, double Y)
{
	return (std::abs(X) < FLT_EPSILON) && (std::abs(Y) < FLT_EPSILON);
}

inline bool IsInnerCorner(double X, double Y)
{
	return (std::abs(X) < FLT_EPSILON) && (Y > FLT_EPSILON) || (std::abs(Y) < FLT_EPSILON) && (X > FLT_EPSILON);
}

inline bool IsTopPart(double X, double Y)
{
	return std::abs(Y - Y_max) < FLT_EPSILON && X <= 0.0;
}

inline bool IsBottomPart(double X, double Y)
{
	return std::abs(Y - Y_min) < FLT_EPSILON && (X <= 1.0 && X >= -1.0);
}

inline bool IsLeftPart(double X, double Y)
{
	return std::abs(X - X_min) < FLT_EPSILON && (Y <= 1.0 && Y >= -1.0);
}

inline bool IsRightPart(double X, double Y)
{
	return std::abs(X - X_max) < FLT_EPSILON && Y <= 0.0;
}

inline bool IsEdge(double X, double Y)
{
	bool result = IsInnerCorner(X, Y)
		|| IsTopPart(X, Y)
		|| IsRightPart(X, Y)
		|| IsLeftPart(X, Y)
		|| IsBottomPart(X, Y);
	return result;
}

inline bool IsOuterCorner(double X, double Y)
{
	bool bLeft = IsLeftPart(X, Y);
	bool bRight = IsRightPart(X, Y);
	bool bTop = IsTopPart(X, Y);
	bool bBottom = IsBottomPart(X, Y);
	return  bLeft && bTop
		|| IsInnerCorner(X, Y) && (bTop || bRight)
		|| bRight && bBottom
		|| bBottom && bLeft;
}

inline bool IsOutOfDomain(double X, double Y)
{
	return (X > FLT_EPSILON) && (Y > FLT_EPSILON);
}


void CreateWithCOO(CSRMatrix& A, std::vector<double>& F, int m, int n, const Domain& D)
{
	int Nn = n + 1, Mn = m + 1; // grid nodes count

	double X_step, Y_step, a = 0.0, b = 0.0;
	X_step = (D.x_max - D.x_min) / (double)m;
	Y_step = (D.y_max - D.y_min) / (double)n;
	double OneBy_h1 = 1.0 / (X_step * X_step); // 1.0/h_1^2 actually
	double OneBy_h2 = 1.0 / (Y_step * Y_step); // 1.0/h_2^2 actually
	double EPS = std::max(X_step * X_step, Y_step * Y_step);

	std::vector<double> A_mat(Mn * Nn, 0.0), B_mat(Mn * Nn, 0.0);
	std::vector<Triplet> COO; // coordinate list matrix format
	COO.reserve(8 * Nn * Mn); // expected COO size

	F.clear();
	F.resize(Nn * Mn, 0.0);

	for (int i = 0; i < Mn; i++) // from Negative X to positive
	{
		double X_cur = D.x_min + i * X_step;

		for (int j = 0; j < Nn; j++) // from negative Y to positive
		{
			double Y_cur = D.y_min + j * Y_step;
			// default assumption - inside D
			a = b = 1.0;
			F[i * Nn + j] = 1.0;

			if (IsEdge(X_cur, Y_cur)) // Edge
			{
				F[i * Nn + j] = 0.5;

				if (IsOuterCorner(X_cur, Y_cur))
				{
					a = b = 0.5 * (1.0 / EPS + 1.0);
					F[i * Nn + j] = 0.25;
				}
				else if (IsTopPart(X_cur, Y_cur) || IsBottomPart(X_cur, Y_cur) || (std::abs(Y_cur) < EPS && !IsLeftPart(X_cur, Y_cur))) // horizontal parts of edge
				{
					a = 0.5 * (1.0 / EPS + 1.0);
					b = 1.0;
				}
				else if (IsLeftPart(X_cur, Y_cur) || IsRightPart(X_cur, Y_cur) || (std::abs(X_cur) < EPS && !IsBottomPart(X_cur, Y_cur))) // vertical parts of edge
				{
					a = 1.0;
					b = 0.5 * (1.0 / EPS + 1.0);
				}

				if (IsLeftPart(X_cur, Y_cur))
				{
					a = 1 / EPS;
				}
				if (IsBottomPart(X_cur, Y_cur))
				{
					b = 1 / EPS;
				}

			}
			else if (IsOutOfDomain(X_cur, Y_cur)) // Outside
			{
				a = b = 1.0 / EPS;
				F[i * Nn + j] = 0.0;
			}
			else if (IsCenter(X_cur, Y_cur))
			{
				F[i * Nn + j] = 0.75;
			}

			if (i > 0)
			{
				COO.push_back({ (i - 1) * Nn + j ,(i - 1) * Nn + j,  a * OneBy_h1 });
				COO.push_back({ (i - 1) * Nn + j ,i * Nn + j,  -a * OneBy_h1 });
				COO.push_back({ i * Nn + j ,(i - 1) * Nn + j,  -a * OneBy_h1 });
			}

			if (j > 0)
			{
				COO.push_back({ i * Nn + j - 1 ,i * Nn + j - 1,  b * OneBy_h2 });
				COO.push_back({ i * Nn + j - 1 ,i * Nn + j,  -b * OneBy_h2 });
				COO.push_back({ i * Nn + j ,i * Nn + j - 1,  -b * OneBy_h2 });
			}

			COO.push_back({ i * Nn + j ,i * Nn + j,  a * OneBy_h1 + b * OneBy_h2 });

			A_mat[i * Nn + j] = a;
			B_mat[i * Nn + j] = b;
		}
	}

	/*std::string AFileName = "Amat.txt";
	PrintFlatMatrix(AFileName, A_mat, Nn, Mn);
	std::string BFileName = "Bmat.txt";
	PrintFlatMatrix(BFileName, B_mat, Nn, Mn);*/

	A = CSRMatrix::COO_To_CSR(COO, Nn * Mn, Nn * Mn);
}

void CreateMatrixesV7(CSRMatrix& A, std::vector<double>& F, int m, int n, const Domain& D) // Variant 7
{
	CreateWithCOO(A, F, m, n, D);
}

void CreateFromLaplace(CSRMatrix& A, std::vector<double>& F, int m, int n)
{
	int Nn = n + 1, Mn = m + 1; // grid nodes count

	double X_step, Y_step, a = 0.0, b = 0.0;
	X_step = (X_max - X_min) / (double)m;
	Y_step = (Y_max - Y_min) / (double)n;
	double OneBy_h1 = 1.0 / (X_step * X_step); // 1.0/h_1^2 actually
	double OneBy_h2 = 1.0 / (Y_step * Y_step); // 1.0/h_2^2 actually
	double EPS = std::max(X_step * X_step, Y_step * Y_step);

	std::vector<double> Amatrix(Nn * Mn, 0.0), Bmatrix(Nn * Mn, 0.0); // matrixes of Laplacian coefficients

	F.clear();
	F.resize(Nn * Mn, 0.0);

	for (int i = 0; i < Mn; i++) // from Negative X to positive
	{
		double X_cur = X_min + i * X_step;

		for (int j = 0; j < Nn; j++) // from negative Y to positive
		{
			double Y_cur = Y_min + j * Y_step;
			// default assumption - inside D
			a = b = 1.0;
			F[i * Nn + j] = 1.0;

			if (IsEdge(X_cur, Y_cur)) // Edge
			{
				if (IsOuterCorner(X_cur, Y_cur))
				{
					a = b = 0.5 * (1.0 / EPS + 1.0);
				}
				else if (IsTopPart(X_cur, Y_cur) || IsBottomPart(X_cur, Y_cur) || (std::abs(Y_cur) < EPS && !IsLeftPart(X_cur, Y_cur)))
				{
					a = 0.5 * (1.0 / EPS + 1.0);
					b = 1.0;
				}
				else if (IsLeftPart(X_cur, Y_cur) || IsRightPart(X_cur, Y_cur) || (std::abs(X_cur) < EPS && !IsBottomPart(X_cur, Y_cur)))
				{
					a = 1.0;
					b = 0.5 * (1.0 / EPS + 1.0);
				}

				if (IsLeftPart(X_cur, Y_cur))
				{
					a = 1 / EPS;
				}
				if (IsBottomPart(X_cur, Y_cur))
				{
					b = 1 / EPS;
				}

				F[i * Nn + j] = 0.5;
			}
			else if (IsOutOfDomain(X_cur, Y_cur)) // Outside
			{
				a = b = 1.0 / EPS;
				F[i * Nn + j] = 0.0;
			}
			Amatrix[i * Nn + j] = a * OneBy_h1;
			Bmatrix[i * Nn + j] = b * OneBy_h2;
		}
	}

	// Corners
	F[0] = F[Nn - 1] = F[Mn / 2 * Nn + Nn - 1] = F[(Mn - 1) * Nn] = F[Mn * Nn - Nn / 2 - 1] = 0.25;

	// Center point
	F[Mn / 2 * Nn + Nn / 2] = 0.75;

	A = CSRMatrix::Laplace_to_CSR(Amatrix, Bmatrix, Mn, Nn);
}


void CreateMatrixesV7(std::vector<CSRMatrix>& A, std::vector<std::vector<double>>& F,
	int m, int n, const std::vector<Domain>& Domains)
{
	assert(A.size() == F.size() && A.size() == Domains.size());

	int i = 0;
	
	for (const auto& D : Domains)
	{
		CreateWithCOO(A[i], F[i], m, n, D);
		i++;
	}
}