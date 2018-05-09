#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


float vec_dot_vec(float *v1, float *v2, int n) {
    float res=0.0;
    for (int i=0; i < n; i+=4) {
        res += v1[i] * v2[i];
    }
    return res;
}

void mat_dot_vec(int N, float **I_kernel, float *V, float *R) {
    for (int i=0; i < N; i++) {
        R[i] = 0;
        for (int j=0; j < N; j++) {
            R[i] += I_kernel[i][j]*V[j];
        }
    }
}


__global__ void mat_dot_vec_cu(int N, float **I_kernel, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = 0;
        for (int j=0; j < N; j++) {
            R[i] += I_kernel[i][j]*V[j];
        }
    }
}

void mat_dot_vec2(int N, float *I_kernel, float *V, float *R) {
    int i, j;
    for (int I=0; I < N*N; I++) {
        j = I % N; 
        i = (I - j) / N;
        R[i] += I_kernel[I]*V[j];
    }
}

__global__ void mat_dot_vec_cu2(int N, float *I_kernel, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int i, j;
    for (int I=index_x; I < N*N; I+=stride_x) {
        j = I % N; 
        i = (I - j) / N;
        R[i] += I_kernel[I]*V[j];
    }
}



int main() {
    int N = 19 * 19 * 19;
    printf("N = %d\n", N);
    float *a;
    float *a_cu;
    float *R;
    float *R_cu;
    float **b;
    float *b2;
    float **b_cu;
    float *b_cu2;
    a = (float *) malloc(N * sizeof(a));
    cudaMallocManaged(&a_cu, N * sizeof(a_cu));
    R = (float *) malloc(N * sizeof(R));
    cudaMallocManaged(&R_cu, N * sizeof(R_cu));
    b = (float **) malloc(N * sizeof(b));
    b2 = (float *) malloc(N * N * sizeof(b2));
    cudaMallocManaged(&b_cu, N * sizeof(b_cu));
    cudaMallocManaged(&b_cu2, N * N * sizeof(b_cu));
    for (int i=0; i < N; i++){
        b[i] = (float *) malloc(N * sizeof(b[i]));
        cudaMallocManaged(&b_cu[i], N * sizeof(b_cu[i]));
    }
    for (int I=0; I < N*N; I++) b_cu2[I] = 1;
    for (int i=0; i < N; i++) {
        a[i] = 1;
        a_cu[i] = 1;
        R[i] = 0;
        R_cu[i] = 0;
    }
    for (int i=0; i < N; i++)
        for (int j=0; j < N; j++) {
            b[i][j] = 0;
            b_cu[i][j] = 0;
            b2[j + N * i] = 0;
            b_cu2[j + N * i] = 0;
            if (i == j) {
                b[i][j] = 1;
                b_cu[i][j] = 1;
                b2[j + N * i] = 1;
                b_cu2[j + N * i] = 1;
            }
        }

    float begin, time_spend;

    begin = clock();
    mat_dot_vec_cu<<<N, N/32>>>(N, b_cu, a_cu, R_cu); cudaDeviceSynchronize();
    time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent with parallelization: %f\n", time_spend);

    //begin = clock();
    //mat_dot_vec_cu2<<<1, 10>>>(N, b_cu2, a_cu, R_cu); cudaDeviceSynchronize();
    //time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    //printf("Time spent with parallelization: %f\n", time_spend);

    //begin = clock();
    //dim3 dimBlock(1, 1);
    //dim3 dimGrid(N, N);
    //mat_dot_vec_cu2<<<dimGrid, dimBlock>>>(N, b_cu, a_cu, R_cu); cudaDeviceSynchronize();
    //time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    //printf("Time spent without parallelization: %f\n", time_spend);

    //begin = clock();
    //mat_dot_vec(N, b, a, R);
    //time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    //printf("Time spent without parallelization: %f\n", time_spend);

    begin = clock();
    mat_dot_vec2(N, b2, a, R);
    time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent without parallelization: %f\n", time_spend);

    float sum = 0;
    for (int i=0; i < N; i++){
        float tmp = R[i] - R_cu[i];
        if (tmp > 0) sum += tmp;
        else sum -= tmp;
    }
    printf("difference: %f\n", sum);
    printf("done");
    //for (int i=0; i < N; i++) printf("%f\n", R[i]);
}
