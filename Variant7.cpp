#include "Variant7.h"
#include <numeric>
#include <limits>

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

void CreateMatrixesV7(CSRMatrix& A, std::vector<double>& f, __int64 M, __int64 N) // Variant 7
{
	__int64 Nn = N + 1, Mn = M + 1; // grid nodes count

	double X_step, Y_step, a = 0.0, b = 0.0;
	X_step = (X_max - X_min) / (double)M;
	Y_step = (Y_max - Y_min) / (double)N;
	double OneBy_h1 = 1.0 / (X_step * X_step); // 1.0/h_1^2 actually
	double OneBy_h2 = 1.0 / (Y_step * Y_step); // 1.0/h_2^2 actually
	double EPS = std::max(X_step * X_step, Y_step * Y_step);

	std::vector<Triplet> COO; // coordinate list matrix format
	COO.reserve(8 * Nn * Mn); // expected COO size

	f.clear();
	f.resize(Nn * Mn, 0.0);

	for (int i = 0; i <= M; i++) // from Negative X to positive
	{
		double X_cur = X_min + i * X_step;

		for (int j = 0; j <= N; j++) // from negative Y to positive
		{
			double Y_cur = Y_min + j * Y_step;
			// default assumption - inside D
			a = b = 1.0;
			f[i * Nn + j] = 1.0;

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

				f[i * Nn + j] = 0.5;
			}
			else if (IsOutOfDomain(X_cur, Y_cur)) // Outside
			{
				a = b = 1.0 / EPS;
				f[i * Nn + j] = 0.0;
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
		}
	}

	// Corners
	f[0] = f[Nn - 1] = f[Mn / 2 * Nn + Nn - 1] = f[(Mn - 1) * Nn] = f[Mn * Nn - Nn / 2 - 1] = 0.25;

	// Center point
	f[Mn / 2 * Nn + Nn / 2] = 0.75;

	A = CSRMatrix::COO_To_CSR(COO, Nn * Mn, Nn * Mn);
}