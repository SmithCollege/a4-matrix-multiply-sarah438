#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double get_clock() {
  struct timeval tv;
  int ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) {
        printf("gettimeofday error");
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void MatrixMulOnHost(float* M, float* N, float* P, int Width){
        for (int i = 0; i < Width; ++i){
                for (int j = 0; j < Width; ++j) {
                        float sum = 0;
                        for (int k = 0; k < Width; ++k) {
                                float a = M[i * Width + k];
                                float b = N[k * Width + j];
                                sum += a * b;
                        }
                        P[i * Width + j] = sum;
                 }
        }
}

int main(){
        double t0 = get_clock(); // inital time

        int size = 10;

        // allocate memory
        float*m = malloc(sizeof(float)*size*size);
        float*n = malloc(sizeof(float)*size*size);
        float*p = malloc(sizeof(float)*size*size);

        for (int i = 0; i<size; i++){
                for(int j=0; j<size;j++){
                        m[i*size+j]=1;
                        n[i*size+j]=1;
                        //printf("[%f][%f] \n", m[i*size+j], n[i*size+j]);
                }
        }

        MatrixMulOnHost(m, n, p, size);

        for(int i=0; i<size;i++){
                for(int j=0; j<size;j++){
                //printf("%f ", p[i*size+j][]);
                if(p[i*size+j]>size){
                        printf("Error at p[%d][%d]: %f\n",i,j,p[i*size+j]);
                }
                }
                printf("\n");
        }
        free(m);
        free(n);
        free(p);

        double t1 = get_clock();
        printf("time per call: %f s\n", (t1-t0));

        return 0;
}
