#pragma once
#include "CSRMatrix.h"

/*
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

void CreateMatrixesV7(CSRMatrix& A, std::vector<double>& f, __int64 M, __int64 N);