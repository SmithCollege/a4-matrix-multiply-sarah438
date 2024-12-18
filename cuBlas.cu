#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#define SIZE 100

// Got help from Glenvelis and used https://www.javaatpoint.com/how-to-add-matrix-in-c

double get_clock(){
        struct timeval tv;
        int ok = gettimeofday(&tv,(void*)0);
        if (ok<0){
                printf("gettimeofday error");
        }
        return (tv.tv_sec * 1.0 + tv.tv_usec *1.0E-6);
}

int main(){
        double t0 = get_clock();
        cublasHandle_t handle;
        cublasCreate(&handle);

        const float a = 1.0f;
        const float b = 0.0f;

        int size = 100;

        float *m;
        float *n;
        float *p;

        cudaMallocManaged(&m, SIZE*sizeof(float) * size*size);
        cudaMallocManaged(&n, SIZE*sizeof(float) * size*size);
        cudaMallocManaged(&p, SIZE*sizeof(float) * size*size);

        for(int i = 0; i <size; i++){
                for(int j = 0; j < size; j++){
                        m[i*size+j] = 1;
                        n[i*size+j] = 1;
                }
        }
        //call cublas
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size,size, size,&a, m, size, n, size, &b, p, size);

        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

        // sync
        cudaDeviceSynchronize();

        for(int i=0; i<size;i++){
                for(int j=0; j< size; j++){
                        //printf("%f ", p[i*size+j]);
                        if(p[i*size+j] > size){
                                printf("Error at p[%d][%d]: %f\n",i,j,p[i*size+j]);
                        }
                }
                printf("\n");
        }

        // free memory
        cudaFree(m);
        cudaFree(n);
        cudaFree(p);

        double t1 = get_clock();
        printf("time per call: %f s\n", (t1-t0));

        return 0;
}
