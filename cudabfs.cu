#include <stdio.h>
#include "graph.h"

typedef unsigned int uint;

__global__
void assignColors(uint *row_ptr, int *col_ind, uint *results, int nov) {
	const uint vIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIndex < nov && results[vIndex] == 0) {
		uint selectedColor = 1;
		for (size_t i = row_ptr[vIndex]; i < row_ptr[vIndex+1]; i++) { // distance-1 neighbor loop
			uint d1Index = col_ind[i];
			if (selectedColor == results[d1Index]) {
				selectedColor++;
				i = row_ptr[vIndex] - 1; // reset the loop
				continue;
			}
			for (size_t j = row_ptr[d1Index]; j < row_ptr[d1Index+1]; j++) {
				uint d2Index = col_ind[j];
				if (selectedColor == results[d2Index]) {
					i = row_ptr[vIndex] - 1; // reset the loop
					break;
				}
			}
		}

	}
}
// && results[vIndex] != 0
__global__
void detectConflicts(uint *row_ptr, int *col_ind, uint *results, int nov, bool * errorCode) {
	const uint vIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIndex < nov) {
		bool secondDistanceConflictFound = false;
		for (size_t i = row_ptr[vIndex]; i < row_ptr[vIndex+1] && !secondDistanceConflictFound; i++) { // distance-1 neighbor loop
			uint d1Index = col_ind[i];
			if (results[vIndex] == results[d1Index]) {
				if (vIndex < d1Index) {
					results[vIndex] = 0;
				}
				else {
					results[d1Index] = 0;
				}
				break;
			}
			for (size_t j = row_ptr[d1Index]; j < row_ptr[d1Index+1] && !secondDistanceConflictFound; j++) {
				uint d2Index = col_ind[j];
				if (results[vIndex] == results[d2Index]) {
					if (vIndex < d2Index) {
						results[vIndex] = 0;
					}
					else {
						results[d2Index] = 0;
					}
					secondDistanceConflictFound = true;
				}
			}
		}
		if (results[vIndex] == 0) {
			*errorCode = true;
		}
	}
}