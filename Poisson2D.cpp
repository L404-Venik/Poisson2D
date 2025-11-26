#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cassert>
#include <omp.h>
#include <mpi.h>

#include "CSRMatrix.h"
#include "Variant7.h"
#include "ConjugateGradient.h"
#include "MPINode.h"


size_t GetMilisecondsCount()
{
	using namespace std::chrono;
	return  duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// Rectangle split to N parts
std::vector<Domain> SplitDomain2D(int P, int& X_segments, int& Y_segments)
{
	if(P <= 0)
		throw std::invalid_argument("P should be positive even number");

	if (P == 1)
	{
		X_segments = Y_segments = 1;
		return std::vector<Domain>{{X_min, X_max, Y_min, Y_max}};
	}

	if (P % 2 == 1)
		throw std::invalid_argument("P should be positive even number");

	std::vector<Domain> subdomains;
	subdomains.reserve(P);

	int best_nx = 1, best_ny = P;
	double best_ratio = 10.0;

	// optimal n_x, n_y
	for (int i = 0; i <= std::log2(P); ++i)
	{
		int nx = 1 << i;
		int ny = P / nx;
		double ratio = static_cast<double>(nx) / ny;

		if (ratio >= 0.5 && ratio <= 2.0)
		{
			double deviation = std::fabs(std::log2(ratio));
			if (deviation < best_ratio)
			{
				best_ratio = deviation;
				best_nx = nx;
				best_ny = ny;
			}
		}
	}

	X_segments = best_nx;
	Y_segments = best_ny;

	double dx = (X_max - X_min) / best_nx;
	double dy = (Y_max - Y_min) / best_ny;

	for (int j = 0; j < best_ny; ++j)
	{
		for (int i = 0; i < best_nx; ++i)
		{
			Domain s;
			s.x_min = X_min + i * dx;
			s.x_max = X_min + (i + 1) * dx;
			s.y_min = Y_min + j * dy;
			s.y_max = Y_min + (j + 1) * dy;
			subdomains.push_back(s);
		}
	}

	return subdomains;
}

void OMPTest()
{
	int N, M; // X axis partitioned to M segments, Y - to N
	int NumThreads;
	M = 800;
	N = 1200;
	NumThreads = 1;

	//std::cin >> M >> N >> NumThreads;
	std::cout << "OpenMP test with " << M << "x" << N << " grid and " << NumThreads << " threads" << std::endl;

	omp_set_num_threads(NumThreads);

	CSRMatrix A;
	std::vector<double> F, omega; // these are matrixes, just flatten
	CreateMatrixesV7(A, F, M, N);

	size_t avgTime = 0;
	int Passes = 3;
	for (int i = 0; i < Passes; i++)
	{
		size_t start = GetMilisecondsCount();
		omega = ConjugateGradient(A, F);
		size_t end = GetMilisecondsCount();

		std::cout << i << " run - " << end - start << " ms" << std::endl;

		avgTime += end - start;
	}
	avgTime /= Passes;

	std::cout << "average " << avgTime << " ms" << std::endl;

	/*std::string ResultFileName = "Result" + std::to_string(M) + "x" + std::to_string(P) + ".txt";
	std::ofstream ResultFile(ResultFileName);
	PrintFlatMatrix(ResultFile, omega, P + 1, M + 1);
	ResultFile.close();*/
}

void GatherOmega(const std::vector<double>& omega, int M, int N, int X_segments, int Y_segments, bool bSave = false)
{
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Rank of the process
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int local_w = M + 1;
	int local_h = N + 1;
	int local_size = local_w * local_h;

	// --- Gather all sizes to root ---
	std::vector<int> recv_counts(world_size);
	MPI_Gather(&local_size, 1, MPI_INT,
		recv_counts.data(), 1, MPI_INT,
		0, MPI_COMM_WORLD);

	std::vector<int> displs(world_size);
	int offset = 0;

	if (world_rank == 0) 
	{
		for (int i = 0; i < world_size; i++) 
		{
			displs[i] = offset;
			offset += recv_counts[i];
		}
	}

	std::vector<double> gathered;
	if (world_rank == 0)
		gathered.resize(offset);

	// Gather ω from all ranks
	MPI_Gatherv(omega.data(), local_size, MPI_DOUBLE,
		gathered.data(), recv_counts.data(),
		displs.data(), MPI_DOUBLE,
		0, MPI_COMM_WORLD);


	if (world_rank == 0)
	{
		// Global grid resolution
		int global_w = X_segments * (local_w - 1) + 1;
		int global_h = Y_segments * (local_h - 1) + 1;

		std::vector<double> global_omega(global_w * global_h);

		// Reconstruct full grid
		for (int rank = 0; rank < world_size; rank++)
		{
			int i = rank % X_segments;   // column
			int j = rank / X_segments;   // row

			int gx0 = i * (local_w - 1);
			int gy0 = j * (local_h - 1);

			const double* block = &gathered[displs[rank]];

			int w_copy = (i == X_segments - 1) ? local_w : (local_w - 1);
			int h_copy = (j == Y_segments - 1) ? local_h : (local_h - 1);

			for (int y = 0; y < h_copy; y++)
			{
				for (int x = 0; x < w_copy; x++)
				{
					int gx = gx0 + x;
					int gy = gy0 + y;

					global_omega[gy * global_w + gx] = block[y * local_w + x];
				}
			}
		}

		if(bSave)
		{
			std::string ResultFileName = "Result" + std::to_string(M * X_segments) + "x" + std::to_string(N * Y_segments) + ".txt";
			std::ofstream ResultFile(ResultFileName);
			PrintFlatMatrix(ResultFile, global_omega, N * Y_segments + 1, M * X_segments + 1);
			ResultFile.close();
		}
	}
}

void MPITest(int argc, char** argv)
{
	int N, M, NumThreads; // X axis partitioned to M segments, Y - to N
	int X_segments, Y_segments; // Number of segments per axis
	int world_rank, world_size;
	MPINode node;
	// this is matrix, just flatten
	std::vector<double> local_omega;

	M = 100;
	N = 100;
	NumThreads = 1;

	//std::cin >> M >> N >> NumThreads;
	omp_set_num_threads(NumThreads);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Rank of the process
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Total number of processes

	if(world_rank == 0)
	{
		std::cout << "MPI test" << std::endl;
		std::cout << world_size << " nodes with " << NumThreads << " threads in each" << std::endl;
		std::cout << M << "x" << N << " grid" << std::endl;
	}
	/*world_size = 4;
	world_rank = 0;*/

	std::vector<Domain> Domains = SplitDomain2D(world_size, X_segments, Y_segments);
	M /= X_segments;
	N /= Y_segments;

	CreateMatrixesV7(node.A, node.F, M, N, Domains[world_rank]);
	node.BuildNeighborInfo(world_rank, X_segments, Y_segments, M + 1, N + 1);

	size_t avgTime = 0;
	int Passes = 1;
	for (int i = 0; i < Passes; i++)
	{
		size_t start = GetMilisecondsCount();
		local_omega = node.ConjugateGradient();
		size_t end = GetMilisecondsCount();

		if(world_rank == 0)
			std::cout << i << " run - " << end - start << " ms" << std::endl;

		avgTime += end - start;
	}
	avgTime /= Passes;

	if (world_rank == 0)
		std::cout << "average " << avgTime << " ms" << std::endl;

	GatherOmega(local_omega, M, N, X_segments, Y_segments, true);

	std::string ResultFileName = std::to_string(world_rank) + "Result" + std::to_string(M * X_segments) + "x" + std::to_string(N * Y_segments) + ".txt";
	std::ofstream ResultFile(ResultFileName);
	PrintFlatMatrix(ResultFile, local_omega, N + 1, M + 1);
	ResultFile.close();

	MPI_Finalize();
}

//extern int NumThreads;

int main(int argc, char** argv)
{
	//OMPTest();
	MPITest(argc, argv);
}
