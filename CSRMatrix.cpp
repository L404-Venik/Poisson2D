#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <assert.h>
#include "CSRMatrix.h"

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

// Constructors
CSRMatrix::CSRMatrix(int r, int c) : m_iRows(r), m_iCols(c)
{
	row_ptr.resize(r + 1, 0);

	// this Poisson task specific size requirements
	col_index.reserve(5 * r);
	values.reserve(5 * r);
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

CSRMatrix CSRMatrix::Laplace_to_CSR(const std::vector<double>& a, const std::vector<double>& b, int Mn, int Nn)
{
	assert(false, "not implemented");

	int N = Mn * Nn;
	CSRMatrix A(Mn * Nn, Mn * Nn);

	int nnz = 0;

	for (int i = 0; i < Mn; ++i)
	{
		for (int j = 0; j < Nn; ++j)
		{
			int row = i * Nn + j;
			A.row_ptr[row] = nnz;

			double diag = 0.0;

			int sz = A.col_index.size();

			// --- Left neighbor ---
			if (j > 0)
			{
				double bv = b[row]; // same as in your COO code
				A.col_index.push_back(row - 1);
				A.values.push_back(-bv);
				diag += bv;
				nnz++;
			}

			// --- Up neighbor ---
			if (i > 0)
			{
				double av = a[row];
				A.col_index.push_back(row - Nn);
				A.values.push_back(-av);
				diag += av;
				nnz++;
			}

			// --- Diagonal ---
			// sum of all attached coefficients
			double self_a = ((i > 0 ? a[row] : 0.0) + (i < Mn - 1 ? a[row + Nn] : 0.0));
			double self_b = ((j > 0 ? b[row] : 0.0) + (j < Nn - 1 ? b[row + 1] : 0.0));
			diag += self_a + self_b;

			A.col_index.push_back(row);
			A.values.push_back(diag);
			nnz++;

			// --- Right neighbor ---
			if (j < Nn - 1)
			{
				double bv = b[row + 1];
				A.col_index.push_back(row + 1);
				A.values.push_back(-bv);
				nnz++;
			}

			// --- Down neighbor ---
			if (i < Mn - 1)
			{
				double av = a[row + Nn];
				A.col_index.push_back(row + Nn);
				A.values.push_back(-av);
				nnz++;
			}
		}
	}

	A.row_ptr[N] = nnz;

	return A;
}


int NumThreads;
// Sparse matrix-vector multiply: y = A * x
std::vector<double> CSRMatrix::VectorMultiply(const std::vector<double>& x) const
{
	if (x.size() != m_iCols)
		throw std::runtime_error("Dimension mismatch in mat-vec multiplication");

	std::vector<double> y(m_iRows, 0.0);

	#pragma omp parallel for schedule(static)// num_threads(NumThreads)
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

double* CSRMatrix::VectorMultiply(double* x) const
{
	double* y = new double[m_iRows];
	std::memset(y, 0.0, m_iRows * sizeof(double));

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

	return D;
}

double* CSRMatrix::GetDiagonalPtr() const
{
	double* D = new double[m_iRows];

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

	return D; 
}

// Serialization
// Serialize matrix into a contiguous byte buffer
std::vector<char> CSRMatrix::Serialize() const
{
	// Compute total byte size
	size_t total_bytes =
		sizeof(int) * 5 +
		sizeof(double) * values.size() +
		sizeof(int) * col_index.size() +
		sizeof(int) * row_ptr.size();

	std::vector<char> buffer(total_bytes);
	char* ptr = buffer.data();

	auto write_int = [&](int x) {
		memcpy(ptr, &x, sizeof(int));
		ptr += sizeof(int);
		};

	write_int(m_iRows);
	write_int(m_iCols);
	write_int((int)values.size());
	write_int((int)col_index.size());
	write_int((int)row_ptr.size());

	memcpy(ptr, values.data(), sizeof(double) * values.size());
	ptr += sizeof(double) * values.size();

	memcpy(ptr, col_index.data(), sizeof(int) * col_index.size());
	ptr += sizeof(int) * col_index.size();

	memcpy(ptr, row_ptr.data(), sizeof(int) * row_ptr.size());

	return buffer;
}

// Deserialize a matrix from a byte buffer
void CSRMatrix::Deserialize(const std::vector<char>& buffer)
{
	const char* ptr = buffer.data();

	auto read_int = [&]() {
		int x;
		memcpy(&x, ptr, sizeof(int));
		ptr += sizeof(int);
		return x;
		};

	m_iRows = read_int();
	m_iCols = read_int();

	int vsize = read_int();
	int csize = read_int();
	int rsize = read_int();

	values.resize(vsize);
	col_index.resize(csize);
	row_ptr.resize(rsize);

	memcpy(values.data(), ptr, sizeof(double) * vsize);
	ptr += sizeof(double) * vsize;

	memcpy(col_index.data(), ptr, sizeof(int) * csize);
	ptr += sizeof(int) * csize;

	memcpy(row_ptr.data(), ptr, sizeof(int) * rsize);
}

// Other funtions

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

//////////////////////////////////////////////////////////////////////////////
// Other common functions

double DotProduct(const std::vector<double>& x, const std::vector<double>& y)
{
	assert(x.size() == y.size());

	double result = 0.0;
	size_t n = x.size();

	#pragma omp parallel for reduction(+:result) schedule(static)// num_threads(NumThreads)
	for (int i = 0; i < n; i++)
		result += x[i] * y[i];

	return result;
}

double DotProduct(const double* x, const double* y, int n)
{
	double result = 0.0;

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

void PrintFlatMatrix(const std::string& sFileName, const std::vector<double>& matrix, int N, int M)
{
	std::ofstream output(sFileName);
	PrintFlatMatrix(output, matrix, N, M);
	output.close();
}