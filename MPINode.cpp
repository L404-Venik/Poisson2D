#include "MPINode.h"
#include "mpi.h"
#include <assert.h>

NeighborDirection opposite(NeighborDirection d)
{
	switch (d)
	{
	case NeighborDirection::LEFT:  return NeighborDirection::RIGHT;
	case NeighborDirection::RIGHT: return NeighborDirection::LEFT;
	case NeighborDirection::UP:    return NeighborDirection::DOWN;
	case NeighborDirection::DOWN:  return NeighborDirection::UP;
	}
}

void MPINode::ExchangeGhosts(std::vector<double>& v)
{
	std::vector<MPI_Request> requests;
	requests.reserve(m_vNeighbors.size() * 2);

	for (auto& nb : m_vNeighbors)
	{
		int count = nb.sendIdx.size();

		// Pack send buffer
		nb.tmpRecv.resize(count);
		nb.tmpSend.resize(count);

		for (int i = 0; i < count; i++)
			nb.tmpSend[i] = v[nb.sendIdx[i]];

		MPI_Request req1, req2;

		MPI_Irecv(
			nb.tmpRecv.data(), count,
			MPI_DOUBLE, nb.NeighborRank, 100 + (int)nb.direction,
			MPI_COMM_WORLD, &req1);

		MPI_Isend(
			nb.tmpSend.data(), count,
			MPI_DOUBLE, nb.NeighborRank, 100 + (int)opposite(nb.direction),
			MPI_COMM_WORLD, &req2);

		// We need recvBuf after wait, so capture by lambda
		requests.push_back(req1);
		requests.push_back(req2);
	}

	// Wait for all to finish
	MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

	// Copy received ghosts into local vector
	for (auto& nb : m_vNeighbors)
	{
		for (size_t i = 0; i < nb.recvIdx.size(); i++)
			v[nb.recvIdx[i]] = nb.tmpRecv[i];

		nb.tmpSend.clear();
		nb.tmpRecv.clear();
	}
}


std::vector<double> MPINode::ConjugateGradient()
{
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int n = A.m_iRows;
	int M = std::sqrt(n);
	const int max_iter = n * world_size;
	const double delta = 0.01;

	std::vector<double> omega(n, 0.0);
	std::vector<double> r = F;          // r0 = F - A*x = F
	std::vector<double> p;
	std::vector<double> Ap(n, 0.0), z(n, 0.0);
	std::vector<double> D = A.GetDiagonal();

	for (int i = 0; i < n; ++i)
		z[i] = r[i] / D[i];

	p = z;

	if (world_rank == 0)
	{
		std::ofstream os("residual.txt", std::ios::out);
		os.close();
	}


	double rz_local = DotProduct(z, r);
	double rz_old = 0.0;
	MPI_Allreduce(&rz_local, &rz_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	for (int it = 0; it < max_iter; it++)
	{
		ExchangeGhosts(p);

		Ap = A.VectorMultiply(p);


		double pAp_local = DotProduct(p, Ap);
		double pAp = 0.0;
		MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


		double alpha = rz_old / pAp;

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
		{
			omega[i] += alpha * p[i];

			r[i] -= alpha * Ap[i];

			z[i] = r[i] / D[i];
		}

		double rz_new_local = DotProduct(z, r);
		double rz_new = 0.0;
		MPI_Allreduce(&rz_new_local, &rz_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		double global_residual = std::sqrt(rz_new);
		if (world_rank == 0)
		{
			std::ofstream os("residual.txt",std::ios::app);
			os << global_residual << std::endl;
			os.close();
		}

		if (global_residual < delta)
		{
			if (world_rank == 0)
				std::cout << "converged in " << it << " stepts\n";

			break; // converged
		}

		double beta = rz_new / rz_old;

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
		{
			p[i] = z[i] + beta * p[i];
		}

		rz_old = rz_new;
	}

	if (world_rank == 0)
		std::cout << "steps limit (" << max_iter << ") reached" << std::endl;

	return omega;
}

void MPINode::BuildNeighborInfo(int world_rank, int X_segments, int Y_segments, int localNx, int localNy)
{
	m_vNeighbors.clear();
	m_vNeighbors.reserve(4);

	// Compute 2D grid coordinates of this rank
	int ix = world_rank % X_segments;
	int iy = world_rank / X_segments;

	auto addNeighbor = [&](int nx, int ny, NeighborDirection dir)
	{
		// Out of bounds = no neighbor
		if (nx < 0 || nx >= X_segments || ny < 0 || ny >= Y_segments)
			return;

		NeighborInfo nb;
		nb.NeighborRank = ny * X_segments + nx;
		nb.direction = dir;

		// LEFT / RIGHT exchange vertical strips
		if (dir == NeighborDirection::LEFT)
		{
			nb.sendIdx.reserve(localNy);
			nb.recvIdx.reserve(localNy);

			for (int y = 0; y < localNy; y++)
				nb.sendIdx.push_back(y * localNx + 0);
			nb.recvIdx = nb.sendIdx;
			/*for (int y = 0; y < localNy; y++)
				nb.recvIdx.push_back(y * localNx + (localNx - 1));*/
		}
		else if (dir == NeighborDirection::RIGHT)
		{
			nb.sendIdx.reserve(localNy);
			nb.recvIdx.reserve(localNy);

			for (int y = 0; y < localNy; y++)
				nb.sendIdx.push_back(y * localNx + (localNx - 1));
			nb.recvIdx = nb.sendIdx;
			/*for (int y = 0; y < localNy; y++)
				nb.recvIdx.push_back(y * localNx + 0);*/
		}

		// UP / DOWN exchange horizontal strips
		else if (dir == NeighborDirection::UP)
		{
			nb.sendIdx.reserve(localNx);
			nb.recvIdx.reserve(localNx);

			int y = 0;
			for (int x = 0; x < localNx; x++)
				nb.sendIdx.push_back(y * localNx + x);

			nb.recvIdx = nb.sendIdx;
			/*int gy = localNy - 1;
			for (int x = 0; x < localNx; x++)
				nb.recvIdx.push_back(gy * localNx + x);*/
		}
		else if (dir == NeighborDirection::DOWN)
		{
			nb.sendIdx.reserve(localNx);
			nb.recvIdx.reserve(localNx);

			int y = localNy - 1;
			for (int x = 0; x < localNx; x++)
				nb.sendIdx.push_back(y * localNx + x);

			nb.recvIdx = nb.sendIdx;
			/*int gy = 0;
			for (int x = 0; x < localNx; x++)
				nb.recvIdx.push_back(gy * localNx + x);*/
		}

		m_vNeighbors.push_back(nb);
	};

	// Add 4 possible neighbors
	/*addNeighbor(ix - 1, iy, NeighborDirection::LEFT);
	addNeighbor(ix + 1, iy, NeighborDirection::RIGHT);
	addNeighbor(ix, iy - 1, NeighborDirection::UP);
	addNeighbor(ix, iy + 1, NeighborDirection::DOWN);*/
	addNeighbor(ix, iy - 1, NeighborDirection::LEFT);
	addNeighbor(ix, iy + 1, NeighborDirection::RIGHT);
	addNeighbor(ix - 1, iy, NeighborDirection::UP);
	addNeighbor(ix + 1, iy, NeighborDirection::DOWN);
}
