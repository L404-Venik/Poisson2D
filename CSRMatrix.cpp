#include <omp.h>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <assert.h>
#include "CSRMatrix.h"

extern int NumThreads;

CSRMatrix::CSRMatrix(int r, int c) : m_iRows(r), m_iCols(c)
{
	row_ptr.resize(r + 1, 0);

	// this Poisson task specific size requirements
	col_index.reserve(5 * r);
	values.reserve(5 * r);
}

int triplet_cmp(const void* a, const void* b)
{
	const Triplet* ta = static_cast<const Triplet*>(a);
	const Triplet* tb = static_cast<const Triplet*>(b);

	if (ta->row < tb->row) return -1;
	if (ta->row > tb->row) return 1;
	if (ta->col < tb->col) return -1;
	if (ta->col > tb->col) return 1;
	return 0;
}

// Convert COO → CSR
CSRMatrix CSRMatrix::COO_To_CSR(const std::vector<Triplet>& coo, int rows, int cols)
{
	CSRMatrix A(rows, cols);

	// Copy and sort by (row, col)
	std::vector<Triplet> sorted = coo;
	qsort(sorted.data(), sorted.size(), sizeof(Triplet), triplet_cmp);

	// Merge duplicates and fill CSR
	int nnz = 0;
	A.row_ptr[0] = 0;
	int current_row = 0;

	for (size_t k = 0; k < sorted.size(); )
	{
		int i = sorted[k].row;
		int j = sorted[k].col;
		double val = 0.0;

		// accumulate duplicates (same row & col)
		while (k < sorted.size() && sorted[k].row == i && sorted[k].col == j)
		{
			val += sorted[k].value;
			k++;
		}

		// fill skipped rows with row_ptr update
		while (current_row < i)
		{
			A.row_ptr[current_row + 1] = nnz;
			current_row++;
		}

		// store element
		A.values.push_back(val);
		A.col_index.push_back(j);
		nnz++;
	}

	// finalize row_ptr
	while (current_row < rows)
	{
		A.row_ptr[current_row + 1] = nnz;
		current_row++;
	}

	return A;
}

CSRMatrix CSRMatrix::COO_To_CSR(const std::unordered_map<std::pair<int, int>, double, PairHash>& entries, int rows, int cols)
{
	CSRMatrix A(rows, cols);
	A.row_ptr.assign(rows + 1, 0);

	// First pass: count nonzeros per row
	for (auto& kv : entries) 
	{
		int i = kv.first.first;
		A.row_ptr[i + 1]++;
	}

	// Prefix sum to get row_ptr
	for (int i = 0; i < rows; i++) 
		A.row_ptr[i + 1] += A.row_ptr[i];

	// Allocate memory
	int nnz = (int)entries.size();
	A.values.resize(nnz);
	A.col_index.resize(nnz);

	// Second pass: fill data
	std::vector<int> offset = A.row_ptr;
	for (auto& kv : entries) 
	{
		int i = kv.first.first;
		int j = kv.first.second;
		double v = kv.second;

		int idx = offset[i]++;
		A.col_index[idx] = j;
		A.values[idx] = v;
	}

	return A;
}

// Sparse matrix-vector multiply: y = A * x
std::vector<double> CSRMatrix::VectorMultiply(const std::vector<double>& x) const
{
	if (x.size() != m_iCols)
		throw std::runtime_error("Dimension mismatch in mat-vec multiplication");

	std::vector<double> y(m_iRows, 0.0);

#pragma omp parallel for schedule(static)
	for (int i = 0; i < m_iRows; i++)
	{
		double sum = 0.0;
		int start = row_ptr[i];
		int end = row_ptr[i + 1];
		for (int k = start; k < end; k++)
			sum += values[k] * x[col_index[k]];

		y[i] = sum;
	}

	return y;
}

void CSRMatrix::print(std::ofstream& OutStream) const
{
	for (int i = 0; i < m_iRows; i++)
	{
		int start = row_ptr[i];
		int end = row_ptr[i + 1];

		// fill row with zeros
		std::vector<double> row(m_iCols, 0.0);

		// place nonzeros
		for (int k = start; k < end; k++)
		{
			row[col_index[k]] = values[k];
		}

		// print row
		for (int j = 0; j < m_iCols; j++)
		{
			OutStream << std::setw(8) << row[j] << " ";
		}
		OutStream << "\n";
	}
}

std::vector<double> CSRMatrix::GetDiagonal() const
{
	std::vector<double> D(m_iRows, 1.0);

	for (int i = 0; i < m_iRows; ++i)
	{
		for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
		{
			if (col_index[k] == i)
			{
				D[i] = values[k];
			}
		}
	}

	return D; // fallback, если диагональ отсутствует 
}

bool CSRMatrix::is_symmetric(double tol) const
{
	if (m_iRows != m_iCols)
		return false;

	for (int i = 0; i < m_iRows; i++)
	{
		for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++)
		{
			int j = col_index[k];
			double a_ij = values[k];

			// Find A[j,i]
			bool found = false;
			for (int kk = row_ptr[j]; kk < row_ptr[j + 1]; kk++)
			{
				if (col_index[kk] == i)
				{
					if (std::fabs(values[kk] - a_ij) > tol) return false;
					found = true;
					break;
				}
			}
			if (!found && std::fabs(a_ij) > tol)
				return false;
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////
// Other common functions

double DotProduct(const std::vector<double>& x, const std::vector<double>& y)
{
	assert(x.size() == y.size());

	double result = 0.0;
	size_t n = x.size();

#pragma omp parallel for reduction(+:result) schedule(static)
	for (int i = 0; i < n; i++)
		result += x[i] * y[i];

	return result;
}

void PrintFlatMatrix(std::ofstream& output, const std::vector<double>& matrix, int N, int M)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			output << matrix[i * N + j] << ' ';
		}
		output << '\n';
	}
}