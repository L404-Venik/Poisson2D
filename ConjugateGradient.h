#pragma once
#include <algorithm>
#include "CSRMatrix.h"

std::vector<double> ConjugateGradient(const CSRMatrix& A, const std::vector<double>& F);
double* ConjugateGradient(const CSRMatrix& A, const double* F);