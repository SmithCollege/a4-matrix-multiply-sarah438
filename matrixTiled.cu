#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#define SIZE 100
#define TILE_WIDTH 10

double get_clock(){
        struct timeval tv;
        int ok = gettimeofday(&tv,(void *) 0);
        if (ok<0){
                printf("gettimeofday error");
        }
        return (tv.tv_sec*1.0+tv.tv_usec*1.0E-6);
}

__global__ void MatrixMulOnKernel(float * M, float * N, float * P, int Width){
        __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
        __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
        int bx = blockIdx.x; int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        // Identify the row and column of the P element to work on
        int Row = by * TILE_WIDTH + ty;
        int Col = bx * TILE_WIDTH + tx;
        float Pvalue = 0;
        // Loop over the M and N tiles required to compute the P element
        // The code assumes that the Width is a multiple of TILE_WIDTH!
        for (int m = 0; m < Width/TILE_WIDTH; ++m) {
                // Collaborative loading of M and N tiles into shared memory
                subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH+tx];
                subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*Width+Col];
                __syncthreads();
                for (int k = 0; k < TILE_WIDTH; ++k){
                        Pvalue += subTileM[ty][k] * subTileN[k][tx];
                        __syncthreads();
                }
                P[Row*Width+Col] = Pvalue;
}

int main(){
        double t0 = get_clock();
        // inital time

    int size = SIZE;
        float * m;
    float * n;
    float * p;

    // allocate memory
    cudaMallocManaged(&m,SIZE*sizeof(float)*size*size);
    cudaMallocManaged(&n,SIZE*sizeof(float)*size*size);
    cudaMallocManaged(&p,SIZE*sizeof(float)*size*size);

    for (int i = 0; i<size; i++){
        for(int j=0; j<size;j++){
                m[i*size+j]=1;
            n[i*size+j]=1;
            //printf("[%f][%f] \n", m[i*size+j], n[i*size+j]);
        }
    }

        // kernal allocation
        dim3 dimGrid(ceil((1.0*size)/TILE_WIDTH),ceil((1.0*Width)/TILE_WIDTH),1);
        dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
        // (ceil((1.0*size)/TILE_WIDTH)), block(TILE_WIDTH, TILE_WIDTH, 1);

    MatrixMulOnKernel<<<dimGrid,dimBlock>>>(m,n,p,size);

    //print error
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

     // sync
     cudaDeviceSynchronize();

     for(int i=0; i<size;i++){
        for(int j=0; j<size;j++){
                //printf("%f ", p[i*size+j][]);
                if(p[i*size+j]>size){
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
