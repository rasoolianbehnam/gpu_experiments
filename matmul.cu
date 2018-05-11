#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    float **indices;
    float **values;
    int *sizes;
} sparseMat;

typedef struct {
    float *array;
    size_t used;
    size_t size;
} Array;

float rel_error(int n, float *a, float *b) {
    float max_diff = -100000;
    float tmp, tmp1, tmp2, sum;
    for (int i=0; i < n; i++) {
        tmp = a[i] - b[i];
        if (tmp < 0) tmp = -tmp;

        tmp1 = a[i];
        tmp2 = b[i];
        if (tmp1 < 0) tmp1 = -tmp1;
        if (tmp2 < 0) tmp2 = -tmp2;
        sum = tmp1 + tmp1;
        if (sum < 1e-8) sum=1e-8;
        if (tmp / sum > max_diff) max_diff=tmp/sum;
    }
    return max_diff;
}
float sum(int n, float *a) {
    float s = 0;
    for (int i=0; i<n; i++) {
        s += a[i];
    }
    return s;
}
void initArray(Array *a, size_t initialSize) {
    //a->array = (float *)malloc(initialSize * sizeof(float));
    cudaMallocManaged(&(a->array), initialSize * sizeof(float));
    a->used = 0;
    a->size = initialSize;
}

void insertArray(Array *a, float element) {
    // a->used is the number of used entries, 
    //because a->array[a->used++] updates a->used 
    //only *after* the array has been accessed.
    // Therefore a->used can go up to a->size 
    if (a->used == a->size) {
        printf("reallocating\n");
        a->size *= 2;
        a->array = (float *)realloc(a->array, a->size * sizeof(float));
    }
    a->array[a->used++] = element;
}
void freeArray(Array *a) {
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

void square_matrix_to_sparse(int N, float **mat, Array **indices, Array **values) {
    for (int i=0; i<N;i++){
        for(int j=0;j<N;j++) {
            if (mat[i][j]) {
                insertArray(indices[i], (float) j);
                insertArray(values[i], mat[i][j]);
            }
        }
    }
}


void mat_dot_vec(int N, float **I_kernel, float *V, float *R) {
    for (int i=0; i < N; i++) {
        R[i] = 0;
        for (int j=0; j < N; j++) {
            R[i] += I_kernel[i][j]*V[j];
        }
    }
}

void array_dot_vec(int N, Array **indices, Array **values, float *V, float *R) {
    for (int i=0; i < N; i++) {
        R[i] = 0;
        for (int j=0; j < indices[i]->used; j++) {
            R[i] += values[i]->array[j] * V[(int)indices[i]->array[j]];
        }
    }
}

void array_dot_vec2(int N, float **indices, float **values, int *sizes, float *V, float *R) {
    for (int i=0; i < N; i+=1) {
        R[i] = 0;
        for (int j=0; j < sizes[i]; j+=1) {
            R[i] += values[i][j] * V[(int)indices[i][j]];
        }
    }
}
__global__  void array_dot_vec_cu(int N, float **indices, float **values, int *sizes, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = 0;
        for (int j=0; j < sizes[i]; j++) {
            R[i] += values[i][j] * V[(int)indices[i][j]];
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



void mat_dot_vec_linear(int N, float *I_kernel, float *V, float *R) {
    int i, j;
    for (int I=0; I < N*N; I++) {
        j = I % N; 
        i = (I - j) / N;
        R[i] += I_kernel[I]*V[j];
    }
}

__global__ void mat_dot_vec_cu_linear(int N, float *I_kernel, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int i, j;
    for (int I=index_x; I < N*N; I+=stride_x) {
        j = I % N; 
        i = (I - j) / N;
        R[i] += I_kernel[I]*V[j];
    }
}

float vec_dot_vec(int N, float *I_kernel, float *V) {
    float out = 0;
    for (int i=0; i < N; i++) {
        out += I_kernel[i]*V[i];
    }
}
void mat_dot_mat(int N, float **I_kernel, float **V, float **R) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            R[i][j] = 0;
            for (int k=0; k < N; k++) {
                R[i][j] += I_kernel[i][k] * V[k][j];
            }
        }
    }
}

float** create_kernel(int imax, int jmax, int kmax, float w) {
    int n1 = imax+3;
    int n2 = jmax+3;
    int n3 = kmax+3;
    int N = n1*n2*n3;
    float **my_mat;
    cudaMallocManaged(&my_mat, N*sizeof(my_mat));
    for (int i=0; i<N; i++) {
        cudaMallocManaged(&my_mat[i], N*sizeof(my_mat[i]));
    }
    for (int I=0; I<N; I++) {
        int k = I % n3;
        int s1 = (I - k) / n3;
        int j = s1 % n2;
        int i = (s1 - j) / n2;
        if (i >= 1 && j >= 1 && k >= 1 && i < imax+1 &&
        j < jmax+1 && k < kmax+1) {
            for (int j=0; j<N; j++)  {
                my_mat[I][j] = w / 6. *
                (my_mat[I-n2*n3][j] +
                my_mat[I-n2][j] +
                my_mat[I-1][j]);
            }
        }
        else
            for (int j=0; j<0; j++)
                my_mat[I][j] = 0;

        my_mat[I][I+1] += w / 6.;
        my_mat[I][I+n3] += w / 6.;
        my_mat[I][I+n2*n3] += w / 6.;
        my_mat[I][I] += 1 - w;
    }
    return my_mat;
}

sparseMat* dense_to_sparse(int N, float** my_mat) {
    Array **indices;
    Array **values;
    indices = (Array **) malloc(N * sizeof(indices));
    values  = (Array **) malloc(N * sizeof(values));
    for (int i=0; i < N; i++) {
        indices[i] = (Array *) malloc(sizeof(indices[i]));
        values[i] = (Array *) malloc(sizeof(values[i]));
        initArray(indices[i], N);
        initArray(values[i], N);
    }
    //Converting normal matrix to sparse
    //the result would be three arrays
    // indices, values and sizes
    square_matrix_to_sparse(N, my_mat, indices, values);
    float **indices_mardas;
    float **values_mardas;
    int *size_mardas;
    cudaMallocManaged(&indices_mardas, N*sizeof(indices_mardas));
    cudaMallocManaged(&values_mardas, N*sizeof(values_mardas));
    cudaMallocManaged(&size_mardas, N*sizeof(size_mardas));
    for (int i=0; i<N; i++) {
        size_mardas[i] = indices[i]->used;
        cudaMallocManaged(&indices_mardas[i], size_mardas[i]*sizeof(indices_mardas[i]));
        cudaMallocManaged(&values_mardas[i], size_mardas[i]*sizeof(values_mardas[i]));
        for (int j=0; j<size_mardas[i]; j++) {
            indices_mardas[i][j] = indices[i]->array[j];
            values_mardas[i][j] = values[i]->array[j];
        }
    }
    sparseMat *sp = (sparseMat *) malloc(sizeof(sparseMat));
    sp->indices=indices_mardas;
    sp->values=values_mardas;
    sp->sizes=size_mardas;
    return sp;
}

void mardas(int imax, int jmax, int kmax) {
    int n1 = imax+3;
    int n2 = jmax+3;
    int n3 = kmax+3;
    int N = n1*n2*n3;
    float w = 1.68;
    printf("N = %d\n", N);

    float **my_mat = create_kernel(imax, jmax, kmax, w);
//    for (int i=0; i<N; i++) {
//        int num_values = rand() % (N / 10);
//        for (int j=0; j < num_values; j++) {
//            my_mat[i][rand()%N] = 1;
//        }
//    }

    float *my_vector;
    cudaMallocManaged(&my_vector, N * sizeof(float));
    for (int i=0; i < N; i++) {
        my_vector[i] = 1;
    }
    
    sparseMat *sp = dense_to_sparse(N, my_mat);


    float begin, time_spend;

    float *result1;
    float *result2;
    float *result3;
    cudaMallocManaged(&result1, N*sizeof(result1));
    cudaMallocManaged(&result2, N*sizeof(result2));
    cudaMallocManaged(&result3, N*sizeof(result3));

    begin = clock();
    mat_dot_vec(N, my_mat, my_vector, result1);
    time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent with NORMAL NO parallelization: %f\n", time_spend);

    begin = clock();
    array_dot_vec2(N, sp->indices, sp->values, sp->sizes, my_vector, result2);
    time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent with spase NO parallelization: %f\n", time_spend);

    begin = clock();
    array_dot_vec_cu<<<N, 1>>>(N, sp->indices, sp->values, sp->sizes, my_vector, result3); cudaDeviceSynchronize();
    time_spend = (float) (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent with spase WITH parallelization: %f\n", time_spend);
    printf("relative error 1 and 2: %f\n", rel_error(N, result1, result2));
    printf("relative error 1 and 3: %f\n", rel_error(N, result1, result3));
    printf("sum 1: %f\n", sum(N, result1));
    printf("sum 2: %f\n", sum(N, result2));

    printf("done");

    //float **I_kernel = (float **) malloc(N*sizeof *I_kernel);
    //for (int i=0; i<N; i++)
    //    I_kernel[i] = (float *) malloc(N*sizeof *I_kernel[i]);

    //for (int I=0; I<N; I++) {
    //    int k = I % n3;
    //    int s1 = (I - k) / n3;
    //    int j = s1 % n2;
    //    int i = (s1 - j) / n2;
    //    if (i >= 1 && j >= 1 && k >= 1 && i < imax+1 &&
    //    j < jmax+1 && k < kmax+1) {
    //        for (int j=0; j<N; j++)  {
    //            I_kernel[I][j] = w / 6. *
    //            (I_kernel[I-n2*n3][j] +
    //            I_kernel[I-n2][j] +
    //            I_kernel[I-1][j]);
    //        }
    //    }
    //    else
    //        for (int j=0; j<0; j++)
    //            I_kernel[I][j] = 0;

    //    I_kernel[I][I+1] += w / 6.;
    //    I_kernel[I][I+n3] += w / 6.;
    //    I_kernel[I][I+n2*n3] += w / 6.;
    //    I_kernel[I][I] += 1 - w;
    //}
    //float **A = (float **) malloc(N * sizeof(float**));
    //for (int i=0; i < N; i++) A[i] = (float*)malloc(N*sizeof(A[i]));
    //mat_dot_mat(N, I_kernel, I_kernel, A);
}
int main() {
    int imax = 32;
    int jmax = 16;
    int kmax = 16;
 
    mardas(imax, jmax, kmax);
}
