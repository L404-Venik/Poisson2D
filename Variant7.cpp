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

	for (int i = N; i >= 0; i--) // from positive Y to negative
	{
		__int64 idx = N - i;
		double Y_cur = Y_min + i * Y_step;

		for (int j = 0; j <= M; j++) // from negative X to positive
		{
			double X_cur = X_min + j * X_step;
			// default assumption - inside D
			a = b = 1.0;
			f[idx * Mn + j] = 1.0;

			if (IsEdge(X_cur, Y_cur)) // Edge
			{
				if(IsOuterCorner(X_cur, Y_cur))
				{
					a = b = 0.5 * (1.0 / EPS + 1.0);
				}
				else if (IsTopPart(X_cur,Y_cur) || IsBottomPart(X_cur, Y_cur) || (std::abs(Y_cur) < EPS && !IsLeftPart(X_cur, Y_cur)))
				{
					a = 0.5 * (1.0 / EPS + 1.0);
					b = 1.0;
				}
				else if (IsLeftPart(X_cur,Y_cur) || IsRightPart(X_cur, Y_cur) || (std::abs(X_cur) < EPS && !IsBottomPart(X_cur, Y_cur)))
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

				f[idx * Mn + j] = 0.5;
			}
			else if (IsOutOfDomain(X_cur, Y_cur)) // Outside
			{
				a = b = 1.0 / EPS;
				f[idx * Mn + j] = 0.0;
			}

			if (idx > 0)
			{
				COO.push_back({ (idx - 1) * Mn + j ,(idx - 1) * Mn + j,  a * OneBy_h1 });
				COO.push_back({ (idx - 1) * Mn + j ,idx * Mn + j,  -a * OneBy_h1 });
				COO.push_back({ idx * Mn + j ,(idx - 1) * Mn + j,  -a * OneBy_h1 });
			}

			if (j > 0)
			{
				COO.push_back({ idx * Mn + j - 1 ,idx * Mn + j - 1,  b * OneBy_h2 });
				COO.push_back({ idx * Mn + j - 1 ,idx * Mn + j,  -b * OneBy_h2 });
				COO.push_back({ idx * Mn + j ,idx * Mn + j - 1,  -b * OneBy_h2 });
			}

			COO.push_back({ idx * Mn + j ,idx * Mn + j,  a * OneBy_h1 + b * OneBy_h2 });
		}
	}

	// Corners
	f[0] = f[Mn / 2] = f[Nn / 2 * Mn + Mn - 1] = f[(Nn - 1) * Mn] = f[Nn * Mn - 1] = 0.25;

	// Center point
	f[Nn / 2 * Mn + Mn / 2] = 0.75;

	A = CSRMatrix::COO_To_CSR(COO, Nn * Mn, Nn * Mn);
}