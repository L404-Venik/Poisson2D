#pragma once
#include <algorithm>
#include "CSRMatrix.h"

enum class NeighborDirection { LEFT, RIGHT, UP, DOWN };

struct NeighborInfo
{
	int NeighborRank;            // which MPI rank it communicates with
	NeighborDirection direction;
	std::vector<int> sendIdx;    // local indices you send
	std::vector<int> recvIdx;    // local ghost indices to fill
	
	std::vector<double> tmpSend, tmpRecv;
};

class MPINode
{
	void ExchangeGhosts(std::vector<double>& v);

public:

	CSRMatrix A;
	std::vector<double> F;
	std::vector<NeighborInfo> m_vNeighbors;

	std::vector<double> ConjugateGradient();
	void BuildNeighborInfo(int world_rank, int X_segments, int Y_segments, int localNx, int localNy);
};