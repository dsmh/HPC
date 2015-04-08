#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 512
#define ELEMENTS 512

__global__ void sum(int *g_idata, int *g_odata,int num_vec) {
	__shared__ int sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if(i<num_vec)
    	sdata[tid] = g_idata[i];
 	else
      sdata[tid]=0;
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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
  
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_R, bytes);
  clock_t start2 = clock();
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);

  // Number of threads in each thread block
  // bloques
  float blocksize = BLOCK_SIZE;
  int i = ELEMENTS;
  while(i>1){
            int grid=ceil(i/blocksize);
          dim3 dimGrid(grid,1,1);
          // hilos
          dim3 dimBlock(blocksize, 1, 1);
          //	sumatoria<<<dimGrid,dimBlock>>>(d_A,d_R);
          sum<<<dimGrid, dimBlock>>>(d_A, d_R,i);
          cudaDeviceSynchronize();
          // Copy array back to host
          cudaMemcpy(d_A, d_R, bytes, cudaMemcpyDeviceToDevice);
          i=ceil(i/blocksize);
  }
  cudaMemcpy(R, d_R, bytes, cudaMemcpyDeviceToHost);
  clock_t end2 = clock();
  double elapsed_seconds2 = end2 - start2;
  printf("Tiempo algoritmo paralelo: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));
  
  
  

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
