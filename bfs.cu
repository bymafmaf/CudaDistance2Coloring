#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <climits>

#include "graphio.h"
#include "graph.h"

const int N = 1024;

char gfile[2048];

using namespace::std;

void usage(){
  printf("./bfs <filename> <sourceIndex>\n");
  exit(0);
}
// CUDA STARTS



__global__
void run(unsigned int *row_ptr, unsigned int *col_ind, unsigned int *results, int nov, unsigned int *errorCode);


/*
You can ignore the ewgths and vwghts. They are there as the read function expects those values
row_ptr and col_ind are the CRS entities. nov is the Number of Vertices
*/

int main(int argc, char *argv[]) {
  unsigned int *row_ptr;
  int *col_ind;
  ewtype *ewghts;
  vwtype *vwghts;
  int nov;

  if(argc != 2)
  usage();

  const char* fname = argv[1];
  strcpy(gfile, fname);

  if(read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0) == -1) {
    printf("error in graph read\n");
    exit(1);
  }
  /****** YOUR CODE GOES HERE *******/
  std::cout << "34 " << row_ptr[34] << '\n';

  unsigned int *d_row_ptr;
  unsigned int *d_col_ind;
  unsigned int *d_results;
  int row_size = nov * sizeof(unsigned int);
  int col_size = (row_ptr[nov] + 1) * sizeof(int);

  unsigned int *results = (unsigned int *) malloc(row_size);
  cudaMalloc((void **)&d_row_ptr, row_size);
  cudaMalloc((void **)&d_col_ind, col_size);
  cudaMalloc((void **)&d_results, row_size);


  for (size_t i = 0; i < nov; i++) {
    for (size_t j = row_ptr[i]; j < row_ptr[i+1]; j++) {
      unsigned int d1index = col_ind[j];
      results[d1index] = row_ptr[i+1] - j + 1;
    }
  }

  for (size_t i = 0; i < nov; i++) {
    printf("%d %d\n", i, results[i]);
  }

  cudaMemcpy(d_results, results, row_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, row_ptr, row_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, col_ind, col_size, cudaMemcpyHostToDevice);

  unsigned int *d_errorCode;
  cudaMalloc((void **)&d_errorCode, sizeof(unsigned int));
  unsigned int *errorCode = (unsigned int*)malloc(sizeof(unsigned int));
  unsigned int *zero = (unsigned int*)malloc(sizeof(unsigned int));
  zero[0] = 0;
  errorCode[0] = 1;
  int k = 0;
  while (errorCode[0] != 0) {
    cudaMemcpy(d_errorCode, zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    run<<<(nov + N -1) / N, N>>>(d_row_ptr, d_col_ind, d_results, nov, d_errorCode);
    k++;
    cudaMemcpy(errorCode, d_errorCode, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  }


  cudaMemcpy(results, d_results, row_size, cudaMemcpyDeviceToHost);

  // TODO use results
  unsigned int max = 0;
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
