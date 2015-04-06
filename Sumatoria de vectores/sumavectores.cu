#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 512
#define ELEMENTS 512

__global__ void sum(int *A, int *C) {
  __shared__ int temp[ELEMENTS];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int local_idx = threadIdx.x;
  temp[local_idx] = A[idx];
  int i = blockDim.x;
  __syncthreads();
  while (i != 0) {
    if (idx + i < ELEMENTS && local_idx < i)
      temp[local_idx] += temp[local_idx + i];
    i /= 2;
    __syncthreads();
  }
  if (local_idx == 0)
    C[blockIdx.x] = temp[0];
}

void sumatoriaSec(int *A, int *r) {
  int value = 0;
  for (int i = 0; i < ELEMENTS; i++) {
    value += A[i];
  }
  *r = value;
}

void llenar(int *A) {
  for (int i = 0; i < ELEMENTS; i++)
    A[i] = 1;
}

int main() {
  size_t bytes = (ELEMENTS) * sizeof(int);
  int *A = (int *)malloc(bytes);
  int *R = (int *)malloc(bytes);
  int s;

  llenar(A);
  // imprimir(A);
  clock_t start = clock();
  sumatoriaSec(A, &s);
  clock_t end = clock();
  double elapsed_seconds = end - start;
  printf("Tiempo algoritmo secuencial: %lf\n",
         (elapsed_seconds / CLOCKS_PER_SEC));

  int *d_A;
  int *d_R;
  clock_t start2 = clock();
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_R, bytes);

  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);

  // Number of threads in each thread block
  // bloques
  float blocksize = BLOCK_SIZE;
  dim3 dimGrid(ceil(ELEMENTS / blocksize), 1, 1);
  // hilos
  dim3 dimBlock(blocksize, 1, 1);

  //	sumatoria<<<dimGrid,dimBlock>>>(d_A,d_R);
  sum << <dimGrid, dimBlock>>> (d_A, d_R);
  // Copy array back to host
  cudaMemcpy(R, d_R, bytes, cudaMemcpyDeviceToHost);
  clock_t end2 = clock();
  double elapsed_seconds2 = end2 - start2;
  printf("Tiempo algoritmo paralelo: %lf\n",
         (elapsed_seconds2 / CLOCKS_PER_SEC));

  if (s != R[0])
    printf(
        "Error al sumar: %d (resultado secuencial) %d (resultado paralelo) \n",
        s, R[0]);
  printf("\n\n RESULTADO SUMA EN PARALELO= %d ", R[0]);
  /*for(int i=0;i<ELEMENTS;i++)
   printf("%d ",R[i]); */
  cudaFree(d_A);
  cudaFree(d_R);
  free(A);
  free(R);
  return 0;
}
