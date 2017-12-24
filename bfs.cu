#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <climits>

#include "graphio.h"
#include "graph.h"

const int N = 1024;

char gfile[2048];

using namespace std;
typedef unsigned int uint;

void usage(){
	printf("./bfs <filename> <sourceIndex>\n");
	exit(0);
}
// CUDA STARTS

void checkCudaError(cudaError_t cudaError, const char * msg = "") {
	if (cudaError != cudaSuccess) {
		cerr << "CUDA error occured [" << msg << "]\nDescription: " << cudaGetErrorString(cudaError) << endl;
		exit(1);
	}
}


__global__
void assignColors(uint *row_ptr, int *col_ind, uint *results, int nov);

__global__
void detectConflicts(uint *row_ptr, int *col_ind, uint *results, int nov, bool *errorCode);

/*
You can ignore the ewgths and vwghts. They are there as the read function expects those values
row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/

int main(int argc, char *argv[]) {
	cudaError_t cudaError;
	// ==== HOST MEMORY ====
	uint *row_ptr;
	int *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	int nov;
	uint * results;
	bool * errorCode = new bool;
	*errorCode = true;
	
	if(argc != 2)
	usage();
	
	const char* fname = argv[1];
	strcpy(gfile, fname);
	
	if(read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1) {
		printf("error in graph read\n");
		exit(1);
	}

	results = (uint *) malloc(nov * sizeof(uint)); // will store color number for each
	for (int i = 0; i < nov; i++) {
		results[i] = 1;
	}
	// ===== DEVICE MEMORY =====
	uint *d_row_ptr;
	int *d_col_ind;
	uint *d_results;
	const size_t row_size = (nov + 1) * sizeof(uint);
	const size_t col_size = (row_ptr[nov]) * sizeof(int);
	bool * d_errorCode; // true, if conflict detected; false otherwise
	
	cudaError = cudaMalloc((void **)&d_errorCode, sizeof(bool));
	checkCudaError(cudaError, "malloc errorCode");
	cudaError = cudaMalloc((void **)&d_row_ptr, row_size);
	checkCudaError(cudaError, "malloc d_row_ptr");
	cudaError = cudaMalloc((void **)&d_col_ind, col_size);
	checkCudaError(cudaError, "malloc d_col_ind");
	cudaError = cudaMalloc((void **)&d_results, nov * sizeof(uint));
	checkCudaError(cudaError, "malloc d_results");
	
	cudaError = cudaMemcpy(d_results, results, nov * sizeof(uint), cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy results");
	cudaError = cudaMemcpy(d_row_ptr, row_ptr, row_size, cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy row_ptr");
	cudaError = cudaMemcpy(d_col_ind, col_ind, col_size, cudaMemcpyHostToDevice);
	checkCudaError(cudaError, "HtoD memcpy col_ind");

	// ==== KERNEL LAUNCH =====
	const uint numBlocks = (nov + N - 1) / N;
	const uint numThreadsPerBlock = N;

	int iterationCounter = 0; // for the following loop
	printf("running kernel with %d blocks with %d threads each\n", numBlocks, numThreadsPerBlock);
	while (*errorCode) { // run kernel until no conflict occurs
		*errorCode = false;
		cudaError = cudaMemcpy(d_errorCode, errorCode, sizeof(bool), cudaMemcpyHostToDevice);
		checkCudaError(cudaError, "memcpy errorCode");
		assignColors<<< numBlocks, numThreadsPerBlock >>>(d_row_ptr, d_col_ind, d_results, nov);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError(), "assignColors() error");
		detectConflicts<<< numBlocks, numThreadsPerBlock >>>(d_row_ptr, d_col_ind, d_results, nov, d_errorCode);
		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError(), "detectConflicts() error");
		cudaError = cudaMemcpy(errorCode, d_errorCode, sizeof(bool), cudaMemcpyDeviceToHost);
		checkCudaError(cudaError);
		iterationCounter++;
		cout << "iteration " << iterationCounter << endl;
	}
	
	cudaError = cudaMemcpy(results, d_results, nov*sizeof(uint), cudaMemcpyDeviceToHost);
	checkCudaError(cudaError, "DtoH memcpy results");
	
	// TODO use results
	uint max = 0;
	for (size_t i = 0; i < nov; i++) {
		if (results[i] > max) {
			max = results[i];
		}
	}
	std::cout << "max is " << max << '\n';
	
	cudaFree(d_row_ptr);
	cudaFree(d_col_ind);
	cudaFree(d_results);
	
	free(row_ptr);
	free(col_ind);
	
	return 1;
}
