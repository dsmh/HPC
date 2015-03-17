
#include<cuda.h>
#include<stdio.h>
#include<time.h>
#include<fstream>

__global__ void kernel(float *d_M, float *d_N, float *d_P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    float Pvalue;
    if((row < width)&&(col < width)){
        Pvalue = 0;
        for (int k = 0; k < width ; ++k){
            Pvalue += d_M[row*width+k] * d_N[k*width+col];
        }
        d_P[row*width+col] = Pvalue;
    }
}

int multHostSec(float *h_M, float *h_N, float *h_P, int width){
    float Pvalue;

    for(int row = 0; row < width ; ++row){
        for(int col = 0; col < width ; ++col){
            Pvalue = 0;
            for(int k = 0; k < width ; ++k){
                Pvalue += h_M[row*width+k] * h_N[k*width+col];
            }
            h_P[row*width+col] = Pvalue;
        }
    }
    return 0;
}

int initMatrix(float *data, int width){
    for(int i = 0; i < width*width; i++)
        data[i] = 13.13;
    return 0;
}



int probarDatos(float *A, float *B, int width){

    for(int i = 0; i < width; ++i){
        for(int j = 0; j < width; ++j){
            if(A[(i*width)+j]!=B[(i*width)+j]){
                printf("Valor diferente\n");
                return 1;
            }
        }
    }
    printf("Matrices iguales\n");
    return 0;
}

int main(){
    float *h_M, *h_N, *h_P,*h_P_d;
    float *d_M, *d_N,*d_P;
    int width = 1024;
    int size = width * width * sizeof(float);
    clock_t start, end, startGPU, endGPU;
    double host_time, device_time;

    h_M = (float*)malloc(size);
    h_N = (float*)malloc(size);
    h_P = (float*)malloc(size);
    h_P_d = (float*)malloc(size);

    if(h_P_d == NULL)
        return 1;

    initMatrix(h_M, width);
    initMatrix(h_N, width);

    /////////INICIO DEL ALGORITMO SECUENCIAL
    start = clock();
    multHostSec(h_M, h_N, h_P, width);
    end = clock();
    host_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    cudaMalloc((void**)&d_M,size);
    cudaMalloc((void**)&d_N,size);
    cudaMalloc((void**)&d_P,size);

    ////////INICIO ALGORITMO PARALELO
    startGPU = clock();
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(width/float(blockSize)),1);
    kernel<<<dimGrid,dimBlock>>>(d_M,d_N,d_P,width);
    cudaDeviceSynchronize();
    cudaMemcpy(h_P_d,d_P,size,cudaMemcpyDeviceToHost);
    endGPU = clock();
    device_time = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo algoritmo secuencial: %.10f\n", host_time);
    printf("Tiempo en paralelo: %.10f\n", device_time);
    printf("La aceleracion fue de:  %.10fX\n",host_time/device_time);
    probarDatos(h_P_d,h_P,width);

    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}




Tiempo algoritmo secuencial: 4,0506970000
Tiempo algoritmo paralelo: 0,0204350000