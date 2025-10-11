#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <assert.h>

struct Triplet 
{
    __int64 row, col;
    double value;
};

class CSRMatrix // Compressed sparse row Matrix class
{
public:
    __int64 m_iRows, m_iCols;
    std::vector<double> values;
    std::vector<__int64> col_index;
    std::vector<__int64> row_ptr;

    CSRMatrix() :m_iRows(0), m_iCols(0) {};
    CSRMatrix(__int64 r, __int64 c);

    // Convert COO → CSR
    static CSRMatrix COO_To_CSR(const std::vector<Triplet>& coo, int rows, int cols);

    // Sparse matrix-vector multiply: y = A * x
    std::vector<double> VectorMultiply(const std::vector<double>& x) const;
    std::vector<double> GetDiagonal() const;

    bool is_symmetric(double tol = 1e-12) const;
    void print(std::ofstream& OutStream) const;
};

double DotProduct(const std::vector<double>& x, const std::vector<double>& y);