#include <stdio.h>
#include "graph.h"

__global__
void run(unsigned int *row_ptr, unsigned int *col_ind, unsigned int *results, int nov, unsigned int *errorCode) {
  const unsigned int vIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vIndex == 0) {
    printf("%s\n", "basla");
    for (size_t i = 0; i < nov; i++) {
      printf("%d %d\n", i, results[i]);
    }
  }
  // if (vIndex < nov) {
  //   unsigned int count = 1;
  //   for (size_t i = row_ptr[vIndex]; i < row_ptr[vIndex+1]; i++) {
  //     unsigned int d1Index = col_ind[i];
  //     if (row_ptr[d1Index+1] < row_ptr[d1Index]) {
  //       printf("d1Index+1 val %d\n", row_ptr[d1Index+1]);
  //       printf("d1Index val %d\n", row_ptr[d1Index]);
  //       printf("d1Index+1 %d\n", d1Index+1);
  //       printf("d1Index %d\n", d1Index);
  //     }
  //     count += row_ptr[d1Index+1] - row_ptr[d1Index] + 1;
  //   }
  //
  //   unsigned int *forbiddenArray = new unsigned int[count];
  //   printf("count %d\n", count);
  //   unsigned int selfColor = results[vIndex];
  //   unsigned int forbiddenIndex = 0;
  //   unsigned int conflictExists = 0;
  //   for (size_t i = row_ptr[vIndex]; i < row_ptr[vIndex+1]; i++) {
  //     unsigned int d1Index = col_ind[i];
  //     forbiddenArray[forbiddenIndex] = results[d1Index];
  //     if (forbiddenArray[forbiddenIndex] == selfColor) {
  //       conflictExists = 1;
  //     }
  //     forbiddenIndex++;
  //     for (size_t j = row_ptr[d1Index]; j < row_ptr[d1Index+1]; j++) {
  //       unsigned int d2Index = col_ind[j];
  //       if (d2Index != vIndex) {
  //         forbiddenArray[forbiddenIndex] = results[d2Index];
  //         if (forbiddenArray[forbiddenIndex] == selfColor) {
  //           conflictExists = 1;
  //         }
  //         forbiddenIndex++;
  //       }
  //     }
  //   }
  //
  //   if (conflictExists == 1) {
  //     for (size_t i = 1; i < nov; i++) {
  //       unsigned int found = 0;
  //       for (size_t j = 0; j < count; j++) {
  //         if (forbiddenArray[j] == i) {
  //           found = 1;
  //           break;
  //         }
  //       }
  //       if (found != 1) {
  //         results[vIndex] = i;
  //         break;
  //       }
  //     }
  //     atomicMax(errorCode, 1);
  //   }
  // }
}
