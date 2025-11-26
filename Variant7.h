#pragma once
#include "CSRMatrix.h"

/*  {(x, y) : −1 < x, y < 1} \ {(x, y) : 0 < x, y < 1}
		domain D  
		   -1.0|
	 **********|**********
	 **********|**********
	 **********|**********
	 **********|**********
	 **********|********** Y
   ------------0-----------→
-1.0 **********|---------- 1.0
	 **********|----------
	 **********|----------
	 **********|----------
	 **********|----------
			1.0↓ X

where '*' - inside D, '-' - outside

*/

constexpr double X_max = 1.0;
constexpr double Y_max = 1.0;
constexpr double X_min = -X_max;
constexpr double Y_min = -Y_max;

struct Domain
{
	double x_min, x_max;
	double y_min, y_max;
};

void CreateMatrixesV7(CSRMatrix& A, std::vector<double>& F, int M, int N, const Domain& D = { X_min, X_max, Y_min, Y_max });

void CreateMatrixesV7(std::vector<CSRMatrix>& A, std::vector<std::vector<double>>& F,
	int M, int N, const std::vector<Domain>& Domains);