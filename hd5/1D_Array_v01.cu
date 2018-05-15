#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define my_index 3

typedef struct {
    float **A;
    float **B;
} Mats;

typedef struct {
    float **indices;
    float **values;
    int *sizes;
} sparseMat;

typedef struct {
    sparseMat *A;
    sparseMat *B;
} sparseMats;

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
float sum2(int n, float **a) {
    float s = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            s += a[i][j];
        }
    }
    return s;
}
void initArray(Array *a, size_t initialSize) {
    cudaMallocManaged(&(a->array ), initialSize * sizeof(float));
    //cudaMallocManaged(&(a->array), initialSize * sizeof(float));
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

void mat_dot_mat(int N, float **I_kernel, float **V, float **R) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            R[i][j] = 0;
            for (int k=0; k < N; k++) R[i][j] += I_kernel[i][k]*V[k][j];
        }
    }
}

__global__ void mat_dot_mat_cu(int N, float **I_kernel, float **V, float **R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        for (int j=0; j < N; j++) {
            R[i][j] = 0;
            for (int k=0; k < N; k++) R[i][j] += I_kernel[i][k]*V[k][j];
        }
    }
}

void mat_add_mat(int N, float **I_kernel, float **V, float **R) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            R[i][j] = I_kernel[i][j]+V[i][j];
        }
    }
}

__global__ void vec_min_vec_cu(int N, float *I_kernel, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = I_kernel[i]-V[i];
    }
}

void vec_min_vec(int N, float *I_kernel, float *V, float *R) {
    for (int i=0; i < N; i++) {
        R[i] = I_kernel[i]-V[i];
    }
}

void array_dot_vec(int N, float **indices, float **values, int *sizes, float *V, float *R) {
    for (int i=0; i < N; i+=1) {
        R[i] = 0;
        for (int j=0; j < sizes[i]; j+=1) {
             R[i] += values[i][j] * V[(int)indices[i][j]];
        }
    }
}

__global__ void array_dot_vec_cu(int N, float **indices, float **values, int *sizes, float *V, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = 0;
        for (int j=0; j < sizes[i]; j+=1) {
            R[i] += values[i][j] * V[(int)indices[i][j]];
        }
    }
}

void special(int N, float **indices1, float **values1, int *sizes1, float *V,  float **indices2, float **values2, int *sizes2, float *g, float *R) {
    for (int i=0; i < N; i+=1) {
        R[i] = 0;
        for (int j=0; j < sizes1[i]; j+=1) {
            R[i] += values1[i][j] * V[(int)indices1[i][j]];
        }
        for (int j=0; j < sizes2[i]; j+=1) {
            R[i] -= values2[i][j] * g[(int)indices2[i][j]];
        }
    }
    for (int i=0; i < N; i+=1) {
        V[i] = R[i];
    }
}
__global__ void special_cu(int N, float **indices1, float **values1, int *sizes1, float *V,  float **indices2, float **values2, int *sizes2, float *g, float *R) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = 0;
        for (int j=0; j < sizes1[i]; j+=1) {
            R[i] += values1[i][j] * V[(int)indices1[i][j]];
        }
        for (int j=0; j < sizes2[i]; j+=1) {
            R[i] -= values2[i][j] * g[(int)indices2[i][j]];
        }
    }
    for (int i=index_x; i < N; i+=stride_x) {
        V[i] = R[i];
    }
}
float** create_kernel(int imax, int jmax, int kmax, float w) {
    int n1 = imax+3;
    int n2 = jmax+3;
    int n3 = kmax+3;
    int N = n1*n2*n3;
    float **my_mat;
    //cudaMallocManaged(&my_mat, N*sizeof(my_mat));
    cudaMallocManaged(&(my_mat ), N*sizeof(my_mat));
    for (int i=0; i<N; i++) {
        //cudaMallocManaged(&my_mat[i], N*sizeof(my_mat[i]));
        cudaMallocManaged(&(my_mat[i] ), N*sizeof(my_mat[i]));
    }
    for (int I=0; I<N; I++) {
        for (int J=0; J<N; J++) my_mat[I][J] = 0;
        int k = I % n3;
        int s1 = (I - k) / n3;
        int j = s1 % n2;
        int i = (s1 - j) / n2;
        if (I % 1000 == 0) printf("%4d / %d\n", I, N);
        if (i >= 1 && i < imax-1 
                && j >= 1 && j < jmax-1 
                && k >= 1 && k < kmax-1) {
            for (int J=0; J<N; J++)  {
                my_mat[I][J] = w / 6. *
                (my_mat[I-n2*n3][J] +
                 my_mat[I-n2][J] +
                 my_mat[I-1][J]);
            }
            my_mat[I][I+1]     += w / 6.;
            my_mat[I][I+n3]    += w / 6.;
            my_mat[I][I+n2*n3] += w / 6.;
            my_mat[I][I]       += 1 - w;
        }
        else
            my_mat[I][I] += 1;

    }
    printf("w = %f\n", w);
    printf("sum of kernel: %f\n", sum2(N, my_mat));
    return my_mat;
}



sparseMat* dense_to_sparse(int N, float** my_mat) {
    Array **indices;
    Array **values;
    cudaMallocManaged(&(indices), N * sizeof(indices));
    cudaMallocManaged(&(values), N * sizeof(values));
    for (int i=0; i < N; i++) {
        cudaMallocManaged(&(indices[i]), sizeof(indices[i]));
        cudaMallocManaged(&(values[i]), sizeof(values[i]));
        initArray(indices[i], N);
        initArray(values[i], N);
    }
    //Converting normal matrix to sparse
    //the result would be three arrays
    // indices, values and sizes
    square_matrix_to_sparse(N, my_mat, indices, values);
    float **indices_t;
    float **values_t;
    int *sizes_t;
    //cudaMallocManaged(&indices_t, N*sizeof(indices_t));
    //cudaMallocManaged(&values_t, N*sizeof(values_t));
    //cudaMallocManaged(&size_t, N*sizeof(size_t));
    cudaMallocManaged(&(indices_t ), N*sizeof(indices_t));
    cudaMallocManaged(&(values_t ), N*sizeof(values_t));
    cudaMallocManaged(&(sizes_t ), N*sizeof(values_t));
    for (int i=0; i<N; i++) {
        sizes_t[i] = indices[i]->used;
        //cudaMallocManaged(&indices_t[i], size_t[i]*sizeof(indices_t[i]));
        //cudaMallocManaged(&values_t[i], size_t[i]*sizeof(values_t[i]));
        cudaMallocManaged(&(indices_t[i] ), sizes_t[i]*sizeof(indices_t[i]));
        cudaMallocManaged(&(values_t[i] ), sizes_t[i]*sizeof(values_t[i]));
        for (int j=0; j<sizes_t[i]; j++) {
            indices_t[i][j] = indices[i]->array[j];
            values_t[i][j] = values[i]->array[j];
        }
    }
    sparseMat *sp;
    cudaMallocManaged(&sp, sizeof(sp));
    sp->indices=indices_t;
    sp->values=values_t;
    sp->sizes=sizes_t;
    return sp;
}

sparseMats* createAB(int N, float **kernel, int iterations, float **A, float **B, float **holder1) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[i][j] = kernel[i][j];
            B[i][j] = 0;
            if (i==j) B[i][j] = 1;
        }
    }
    for (int kk=0; kk < iterations-1; kk++) {
        printf("iteration %d\n", kk);
        mat_add_mat(N, B, A, B);
        //mat_dot_mat(N, kernel, A, holder1);
        mat_dot_mat_cu<<<N, 1>>>(N, kernel, A, holder1); cudaDeviceSynchronize();
        float **tmp = holder1;
        holder1 = A;
        A = tmp;
    }
    sparseMats *AB;
    cudaMallocManaged(&AB, sizeof(AB));
    AB->A = dense_to_sparse(N, A);
    AB->B = dense_to_sparse(N, B);
    printf("sum of A: %f", sum2(N, A));
    return AB;
}

sparseMats* readAB(int n1, int n2, int n3, int iterations, float **A, float **B) {
    char* file_name;
    file_name = (char *) malloc(20*sizeof(file_name));
    sprintf(file_name, "A_%d_%d_%d_%d.txt", n1, n2, n3, iterations);
    printf("file name: %s\n", file_name);
    FILE *fp = fopen(file_name, "r");
    const char s[2] = " ";
    char *token;
    int i;
    int r, k;
    float val;
    float sumA = 0;
    if(fp != NULL) {
        char line[20];
        while(fgets(line, sizeof line, fp) != NULL) {
            token = strtok(line, s);
            r = atof(token);
            //printf("%d ",r);
            token = strtok(NULL,s);
            k = atof(token);
            //printf("%d ",k);
            token = strtok(NULL,s);
            val = atof(token);
            //printf("%f\n",val);
            A[r][k] = val;
            sumA += A[r][k];
        }
        fclose(fp);
    } else {
        perror(file_name);
    } 

    sprintf(file_name, "B_%d_%d_%d_%d.txt", n1, n2, n3, iterations);
    printf("file name: %s\n", file_name);
    fp = fopen(file_name, "r");
    float sumB = 0;
    if(fp != NULL) {
        char line[20];
        while(fgets(line, sizeof line, fp) != NULL) {
            token = strtok(line, s);
            r = atof(token);
            //printf("%d ",r);
            token = strtok(NULL,s);
            k = atof(token);
            //printf("%d ",k);
            token = strtok(NULL,s);
            val = atof(token);
            //printf("%f\n",val);
            B[r][k] = val;
            sumB += B[r][k];
        }
        fclose(fp);
    } else {
        perror(file_name);
    } 

    sparseMats *AB;
    cudaMallocManaged(&AB, sizeof(AB));
    AB->A = dense_to_sparse(n1*n2*n3, A);
    AB->B = dense_to_sparse(n1*n2*n3, B);
    printf("Sum of A: %f\n", sumA);
    printf("Sum of B: %f\n", sumB);
    return AB;
}
void poisson_solve3(int imax, int jmax, int kmax, 
    int n1, int n2, int n3, int N, int iterations, 
    float* V, float* g, float* g_temp, float *R, float w,
    float h, sparseMats* AB, float* holder) {

        for (int I=0; I<N; I++) g_temp[I] = w*h*h*g[I]/6.;
        for (int I=0; I<N; I++) {
            int k = I % n3;
            int s1 = (I - k) / n3;
            int j = s1 % n2;
            int i = (s1 - j) / n2;
            if (i >= 1 && i < imax-1 
                    && j >= 1 && j < jmax-1 
                    && k >= 1 && k < kmax-1) {
                g_temp[I] += w/6.*(g_temp[I-1]+g_temp[I-n3]+g_temp[I-n3*n2]);
            }
            else {
                g_temp[I] = 0;
            }
        }

        //array_dot_vec(N, AB->A->indices, AB->A->values, AB->A->sizes, V, R);
        //array_dot_vec_cu<<<1024, 1024>>>(N, AB->A->indices, AB->A->values, AB->A->sizes, V, R); cudaDeviceSynchronize();
        //array_dot_vec(N, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, holder);
        //array_dot_vec_cu<<<1024, 1024>>>(N, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, holder); cudaDeviceSynchronize();
        //vec_min_vec(N, R, holder, V);
        //vec_min_vec_cu<<<1024, 1024>>>(N, R, holder, V); cudaDeviceSynchronize();
        //special(N, AB->A->indices, AB->A->values, AB->A->sizes, V, AB->B->indices, AB->B->values, AB->B->sizes, g, R);
        special_cu<<<2048*2, 1024>>>(N, AB->A->indices, AB->A->values, AB->A->sizes, V, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, R); cudaDeviceSynchronize();
        //printf("\n");
    }

__global__ void poisson_solve3_cu(int imax, int jmax, int kmax, 
    int n1, int n2, int n3, int N, int iterations, 
    float* V, float* g, float* g_temp, float *R, float w,
    float h, sparseMats* AB, float* holder) {

    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
        for (int I=index_x; I<N; I+=stride_x) g_temp[I] = w*h*h*g[I]/6.;
        for (int I=index_x; I<N; I+=stride_x) {
            int k = I % n3;
            int s1 = (I - k) / n3;
            int j = s1 % n2;
            int i = (s1 - j) / n2;
            if (i >= 1 && i < imax-1 
                    && j >= 1 && j < jmax-1 
                    && k >= 1 && k < kmax-1) {
                g_temp[I] += w/6.*(g_temp[I-1]+g_temp[I-n3]+g_temp[I-n3*n2]);
            }
            else {
                g_temp[I] = 0;
            }
        }
    for (int i=index_x; i < N; i+=stride_x) {
        R[i] = 0;
        for (int j=0; j < AB->A->sizes[i]; j+=1) {
            R[i] += AB->A->values[i][j] * V[(int)AB->A->indices[i][j]];
        }
        for (int j=0; j < AB->B->sizes[i]; j+=1) {
            R[i] -= AB->B->values[i][j] * g[(int)AB->B->indices[i][j]];
        }
    }
    for (int i=index_x; i < N; i+=stride_x) {
        V[i] = R[i];
    }


        //array_dot_vec(N, AB->A->indices, AB->A->values, AB->A->sizes, V, R);
        //array_dot_vec_cu<<<1024, 1024>>>(N, AB->A->indices, AB->A->values, AB->A->sizes, V, R); cudaDeviceSynchronize();
        //array_dot_vec(N, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, holder);
        //array_dot_vec_cu<<<1024, 1024>>>(N, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, holder); cudaDeviceSynchronize();
        //vec_min_vec(N, R, holder, V);
        //vec_min_vec_cu<<<1024, 1024>>>(N, R, holder, V); cudaDeviceSynchronize();
        //special(N, AB->A->indices, AB->A->values, AB->A->sizes, V, AB->B->indices, AB->B->values, AB->B->sizes, g, R);
        //special_cu<<<2048*2, 1024>>>(N, AB->A->indices, AB->A->values, AB->A->sizes, V, AB->B->indices, AB->B->values, AB->B->sizes, g_temp, R); cudaDeviceSynchronize();
        //printf("\n");
    }

void poisson_solve2(int imax, int jmax, int kmax, 
    int n1, int n2, int n3, int N, int iterations, 
    float* V, float* g, float* g_temp, float *R, float w,
    float h, sparseMat* kernel, float** kernel2) {
        for (int I=0; I<N; I++) g_temp[I] = w*h*h*g[I]/6.;
        for (int I=0; I<N; I++) {
            int k = I % n3;
            int s1 = (I - k) / n3;
            int j = s1 % n2;
            int i = (s1 - j) / n2;
            if (i >= 1 && i < imax-1 
                    && j >= 1 && j < jmax-1 
                    && k >= 1 && k < kmax-1) {
                g_temp[I] += w/6.*(g_temp[I-1]+g_temp[I-n3]+g_temp[I-n3*n2]);
            }
            else {
                g_temp[I] = 0;
            }
        }
        for (int kk=0; kk<iterations; kk++) {
            array_dot_vec(N, kernel->indices, kernel->values, kernel->sizes, V, R);
            //array_dot_vec_cu<<<N, 1>>>(N, kernel->indices, kernel->values, kernel->sizes, V, R); cudaDeviceSynchronize();
            //mat_dot_vec(N, kernel2, V, R);
            for (int i=0; i<N; i++) V[i] = R[i] - g_temp[i];
            //printf("%f, ", sum(N, V));
        }
        //printf("\n");
    }

void poisson_solve(int imax, int jmax, int kmax, int n1, int n2, int n3, int N, int iterations, float* V, float* g, float *R, float w, float h) {
      for (int kk=0; kk<iterations; kk++) {
        for (int I = 0; I < N; I ++) {
             int k = I % n3;
             int s1 = (I - k) / n3;
             int j = s1 % n2;
             int i = (s1 - j) / n2;
            if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
            R[k + n3 * (j + n2 * (i))]=
                (V[k + n3 * (j + n2 * (i+1))]+
                     V[k + n3 * (j + n2 * (i-1))]+
                     V[k + n3 * (j+1 + n2 * (i))]+
                     V[k + n3 * (j-1 + n2 * (i))]+
                     V[k+1 + n3 * (j + n2 * (i))]+
                     V[k-1 + n3 * (j + n2 * (i))]
                 ) / 6.0 - V[k + n3 * (j + n2 * (i))]- (h*h)*g[k + n3 * (j + n2 * (i))]/6.0;
            V[k + n3 * (j + n2 * (i))] += w*R[k + n3 * (j + n2 * (i))];
        }
    }
}

void before(int imax, int jmax, int kmax, int tmax, float *ne, float *ni, float *difxne, float *difyne, float *difxni, float *difyni, float *difxyne, float *difxyni, float *Exy, float *fexy, float *fixy, float *g, float* g_temp, float *R, float *Ex, float *Ey, float *fex, float *fey, float *fix, float *fiy, float *V, float *L, float *difzne, float *difzni, float *Ez, float *fez, float *fiz, float qi, float qe, float kr, float ki, float si, float sf, float alpha, float q, float pie, float Ta , float w , float eps0 , float Te, float Ti, float B, float Kb, float me, float mi, float nue, float nui, float denominator_e, float denominator_i, float nn, float dt, float h, float wce, float wci, float mue, float mui, float dife, float difi) {
    int index_x = 0;
    int stride_x = 1;
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1; 
    N=n1*n2*n3;
      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        g[k + n3 * (j + n2 * (i))]=-(ne[k + n3 * (j + n2 * (i))]*qe+ni[k + n3 * (j + n2 * (i))]*qi)/eps0;
    }
 
}
__global__ void before_cu(int imax, int jmax, int kmax, int tmax, float *ne, float *ni, float *difxne, float *difyne, float *difxni, float *difyni, float *difxyne, float *difxyni, float *Exy, float *fexy, float *fixy, float *g, float* g_temp, float *R, float *Ex, float *Ey, float *fex, float *fey, float *fix, float *fiy, float *V, float *L, float *difzne, float *difzni, float *Ez, float *fez, float *fiz, float qi, float qe, float kr, float ki, float si, float sf, float alpha, float q, float pie, float Ta , float w , float eps0 , float Te, float Ti, float B, float Kb, float me, float mi, float nue, float nui, float denominator_e, float denominator_i, float nn, float dt, float h, float wce, float wci, float mue, float mui, float dife, float difi) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1; 
    N=n1*n2*n3;
      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        g[k + n3 * (j + n2 * (i))]=-(ne[k + n3 * (j + n2 * (i))]*qe+ni[k + n3 * (j + n2 * (i))]*qi)/eps0;
    }
 
}
void after(int imax, int jmax, int kmax, int tmax, float *ne, float *ni, float *difxne, float *difyne, float *difxni, float *difyni, float *difxyne, float *difxyni, float *Exy, float *fexy, float *fixy, float *g, float* g_temp, float *R, float *Ex, float *Ey, float *fex, float *fey, float *fix, float *fiy, float *V, float *L, float *difzne, float *difzni, float *Ez, float *fez, float *fiz, float qi, float qe, float kr, float ki, float si, float sf, float alpha, float q, float pie, float Ta , float w , float eps0 , float Te, float Ti, float B, float Kb, float me, float mi, float nue, float nui, float denominator_e, float denominator_i, float nn, float dt, float h, float wce, float wci, float mue, float mui, float dife, float difi) {
    int index_x = 0;
    int stride_x = 1;
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1; 
    N=n1*n2*n3;
  for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax-1 || j >= jmax || k >= kmax) continue;
        Ex[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j + n2 * (i+1))])/h;
        difxne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j + n2 * (i+1))]-ne[k + n3 * (j + n2 * (i))])/h;
        difxni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j + n2 * (i+1))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax-1 || k >= kmax) continue;
        Ey[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j+1 + n2 * (i))])/h;
        difyne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j+1 + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
        difyni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j+1 + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax || k >= kmax-1) continue;
       Ez[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k+1 + n3 * (j + n2 * (i))])/h;
       difzne[k + n3 * (j + n2 * (i))]=(ne[k+1 + n3 * (j + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
       difzni[k + n3 * (j + n2 * (i))]=(ni[k+1 + n3 * (j + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
     }

// -----------------------------------------------------------------------------------------------
       /* Since I am using mid points for Calculating electric field and density gradient,
        to calculate Ex at any point that I don't have it directly, the average over
        the neighboring points is used instead. these average variables are, exy, fexy, fixy, ...*/
        // Calculating the average values of Ex and gradiant_x
   for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ex[k + n3 * (j + n2 * (i))]+Ex[k + n3 * (j+1 + n2 * (i))]+Ex[k + n3 * (j + n2 * (i-1))]+Ex[k + n3 * (j+1 + n2 * (i-1))]) ;
        difxyne[k + n3 * (j + n2 * (i))]=0.25*(difxne[k + n3 * (j + n2 * (i))]+difxne[k + n3 * (j+1 + n2 * (i))]+difxne[k + n3 * (j + n2 * (i-1))]+difxne[k + n3 * (j+1 + n2 * (i-1))]);
        difxyni[k + n3 * (j + n2 * (i))]=0.25*(difxni[k + n3 * (j + n2 * (i))]+difxni[k + n3 * (j+1 + n2 * (i))]+difxni[k + n3 * (j + n2 * (i-1))]+difxni[k + n3 * (j+1 + n2 * (i-1))]);
    }


// -----------------------------------------------------------------------------------------------
        // Here we calculate the fluxes in y direction

       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fey[k + n3 * (j + n2 * (i))]= (-0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*mue*Ey[k + n3 * (j + n2 * (i))]-dife*difyne[k + n3 * (j + n2 * (i))]
        -wce*q*0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(me*nue*nue)-wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue)/denominator_e;
        fiy[k + n3 * (j + n2 * (i))]= (0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*mui*Ey[k + n3 * (j + n2 * (i))]-difi*difyni[k + n3 * (j + n2 * (i))]
        -wci*q*0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui)+wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui)/denominator_i;
    }


    for ( I =index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax-1 || k >= kmax-1) continue;

        if (fey[k + n3 * (0 + n2 * (i))] > 0.0){
                fey[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (0 + n2 * (i))] > 0.0){
                fiy[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fey[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fey[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fiy[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------
        // Calculating the average Exy and difxy to be used in x direction fluxes
       // Calculating the average values of Ey and gradiant_y


      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }


      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ey[k + n3 * (j + n2 * (i))]+Ey[k + n3 * (j-1 + n2 * (i))]+Ey[k + n3 * (j + n2 * (i+1))]+Ey[k + n3 * (j-1 + n2 * (i+1))]);
        difxyne[k + n3 * (j + n2 * (i))]= 0.25*(difyne[k + n3 * (j + n2 * (i))]+difyne[k + n3 * (j-1 + n2 * (i))]+difyne[k + n3 * (j + n2 * (i+1))]+difyne[k + n3 * (j-1 + n2 * (i+1))]);
        difxyni[k + n3 * (j + n2 * (i))]= 0.25*(difyni[k + n3 * (j + n2 * (i))]+difyni[k + n3 * (j-1 + n2 * (i))]+difyni[k + n3 * (j + n2 * (i+1))]+difyni[k + n3 * (j-1 + n2 * (i+1))]);

    }

// -----------------------------------------------------------------------------------------------
        // Now ready to calculate the fluxes in x direction

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fex[k + n3 * (j + n2 * (i))]=(-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])*mue*Ex[k + n3 * (j + n2 * (i))]-dife*difxne[k + n3 * (j + n2 * (i))]
        +wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue+wce*q*0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])/(me*nue*nue)*Exy[k + n3 * (j + n2 * (i))])/denominator_e;
        fix[k + n3 * (j + n2 * (i))]=(0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*mui*Ex[k + n3 * (j + n2 * (i))]-difi*difxni[k + n3 * (j + n2 * (i))]
        -wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui+wci*q*0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui))/denominator_i;
    }


    for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;

        if (j * k == 0 ||  j >= jmax-1 || k >= kmax-1) continue;

        if (fex[k + n3 * (j + n2 * (0))] > 0.0){
                fex[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (0))] > 0.0){
                fix[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fex[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fex[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fix[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------

        // Now we calculate the fluxes in z direction
      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fez[k + n3 * (j + n2 * (i))]=-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k+1 + n3 * (j + n2 * (i))])*mue*Ez[k + n3 * (j + n2 * (i))]-dife*difzne[k + n3 * (j + n2 * (i))];
        fiz[k + n3 * (j + n2 * (i))]=0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k+1 + n3 * (j + n2 * (i))])*mui*Ez[k + n3 * (j + n2 * (i))]-difi*difzni[k + n3 * (j + n2 * (i))];

    }
       // BC on fluxes

    for ( I = index_x; I < n1 * n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1) continue;
        if (fez[0 + n3 * (j + n2 * (i))]>0.0){
            fez[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[0 + n3 * (j + n2 * (i))]>0.0){
            fiz[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fez[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fez[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fiz[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
    }
// -----------------------------------------------------------------------------------------------


       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]-dt*(fex[k + n3 * (j + n2 * (i))]-fex[k + n3 * (j + n2 * (i-1))]+fey[k + n3 * (j + n2 * (i))]-fey[k + n3 * (j-1 + n2 * (i))]+fez[k + n3 * (j + n2 * (i))]-fez[k-1 + n3 * (j + n2 * (i))])/h ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]-dt*(fix[k + n3 * (j + n2 * (i))]-fix[k + n3 * (j + n2 * (i-1))]+fiy[k + n3 * (j + n2 * (i))]-fiy[k + n3 * (j-1 + n2 * (i))]+fiz[k + n3 * (j + n2 * (i))]-fiz[k-1 + n3 * (j + n2 * (i))])/h ;
    }





        for ( I = index_x; I < n1 * n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax || j >= jmax ) continue;

        ne[0 + n3 * (j + n2 * (i))] = -dt*fez[0 + n3 * (j + n2 * (i))]/h ;
        ni[0 + n3 * (j + n2 * (i))] = -dt*fiz[0 + n3 * (j + n2 * (i))]/h ;

    }

     for ( I = index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = -dt*fey[k + n3 * (0 + n2 * (i))]/h ;
        ni[k + n3 * (0 + n2 * (i))] = -dt*fiy[k + n3 * (0 + n2 * (i))]/h ;

    }




      for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= -dt*fex[k + n3 * (j + n2 * (0))]/h ;
       ni[k + n3 * (j + n2 * (0))]= -dt*fix[k + n3 * (j + n2 * (0))]/h ;


    }




        // BC on densities

     for ( I = index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (0 + n2 * (i))] = 0.0 ;

        ne[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;

    }




      for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= 0.0 ;
       ni[k + n3 * (j + n2 * (0))]= 0.0 ;

       ne[k + n3 * (j + n2 * (imax-1))]= 0.0 ;
       ni[k + n3 * (j + n2 * (imax-1))]= 0.0 ;


    }



    for ( I = index_x; I < n1*n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax+1 || j >= jmax+1) continue;
        ne[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ne[0 + n3 * (j + n2 * (i))]=0.0;
        ni[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ni[0 + n3 * (j + n2 * (i))]=0.0;

    }




        // calculating the loss
         sf=0.0;
       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
         if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        sf=sf+ne[k + n3 * (j + n2 * (i))] ;
    }

        alpha=(si-sf)/sf;

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
    }
      // if (myTime%100==0.0){
     //  }

}

__global__ void after_cu(int imax, int jmax, int kmax, int tmax, float *ne, float *ni, float *difxne, float *difyne, float *difxni, float *difyni, float *difxyne, float *difxyni, float *Exy, float *fexy, float *fixy, float *g, float* g_temp, float *R, float *Ex, float *Ey, float *fex, float *fey, float *fix, float *fiy, float *V, float *L, float *difzne, float *difzni, float *Ez, float *fez, float *fiz, float qi, float qe, float kr, float ki, float si, float sf, float alpha, float q, float pie, float Ta , float w , float eps0 , float Te, float Ti, float B, float Kb, float me, float mi, float nue, float nui, float denominator_e, float denominator_i, float nn, float dt, float h, float wce, float wci, float mue, float mui, float dife, float difi) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1; 
    N=n1*n2*n3;
  for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax-1 || j >= jmax || k >= kmax) continue;
        Ex[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j + n2 * (i+1))])/h;
        difxne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j + n2 * (i+1))]-ne[k + n3 * (j + n2 * (i))])/h;
        difxni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j + n2 * (i+1))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax-1 || k >= kmax) continue;
        Ey[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j+1 + n2 * (i))])/h;
        difyne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j+1 + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
        difyni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j+1 + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax || k >= kmax-1) continue;
       Ez[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k+1 + n3 * (j + n2 * (i))])/h;
       difzne[k + n3 * (j + n2 * (i))]=(ne[k+1 + n3 * (j + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
       difzni[k + n3 * (j + n2 * (i))]=(ni[k+1 + n3 * (j + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
     }

// -----------------------------------------------------------------------------------------------
       /* Since I am using mid points for Calculating electric field and density gradient,
        to calculate Ex at any point that I don't have it directly, the average over
        the neighboring points is used instead. these average variables are, exy, fexy, fixy, ...*/
        // Calculating the average values of Ex and gradiant_x
   for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ex[k + n3 * (j + n2 * (i))]+Ex[k + n3 * (j+1 + n2 * (i))]+Ex[k + n3 * (j + n2 * (i-1))]+Ex[k + n3 * (j+1 + n2 * (i-1))]) ;
        difxyne[k + n3 * (j + n2 * (i))]=0.25*(difxne[k + n3 * (j + n2 * (i))]+difxne[k + n3 * (j+1 + n2 * (i))]+difxne[k + n3 * (j + n2 * (i-1))]+difxne[k + n3 * (j+1 + n2 * (i-1))]);
        difxyni[k + n3 * (j + n2 * (i))]=0.25*(difxni[k + n3 * (j + n2 * (i))]+difxni[k + n3 * (j+1 + n2 * (i))]+difxni[k + n3 * (j + n2 * (i-1))]+difxni[k + n3 * (j+1 + n2 * (i-1))]);
    }


// -----------------------------------------------------------------------------------------------
        // Here we calculate the fluxes in y direction

       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fey[k + n3 * (j + n2 * (i))]= (-0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*mue*Ey[k + n3 * (j + n2 * (i))]-dife*difyne[k + n3 * (j + n2 * (i))]
        -wce*q*0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(me*nue*nue)-wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue)/denominator_e;
        fiy[k + n3 * (j + n2 * (i))]= (0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*mui*Ey[k + n3 * (j + n2 * (i))]-difi*difyni[k + n3 * (j + n2 * (i))]
        -wci*q*0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui)+wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui)/denominator_i;
    }


    for ( I =index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax-1 || k >= kmax-1) continue;

        if (fey[k + n3 * (0 + n2 * (i))] > 0.0){
                fey[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (0 + n2 * (i))] > 0.0){
                fiy[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fey[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fey[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fiy[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------
        // Calculating the average Exy and difxy to be used in x direction fluxes
       // Calculating the average values of Ey and gradiant_y


      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }


      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ey[k + n3 * (j + n2 * (i))]+Ey[k + n3 * (j-1 + n2 * (i))]+Ey[k + n3 * (j + n2 * (i+1))]+Ey[k + n3 * (j-1 + n2 * (i+1))]);
        difxyne[k + n3 * (j + n2 * (i))]= 0.25*(difyne[k + n3 * (j + n2 * (i))]+difyne[k + n3 * (j-1 + n2 * (i))]+difyne[k + n3 * (j + n2 * (i+1))]+difyne[k + n3 * (j-1 + n2 * (i+1))]);
        difxyni[k + n3 * (j + n2 * (i))]= 0.25*(difyni[k + n3 * (j + n2 * (i))]+difyni[k + n3 * (j-1 + n2 * (i))]+difyni[k + n3 * (j + n2 * (i+1))]+difyni[k + n3 * (j-1 + n2 * (i+1))]);

    }

// -----------------------------------------------------------------------------------------------
        // Now ready to calculate the fluxes in x direction

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fex[k + n3 * (j + n2 * (i))]=(-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])*mue*Ex[k + n3 * (j + n2 * (i))]-dife*difxne[k + n3 * (j + n2 * (i))]
        +wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue+wce*q*0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])/(me*nue*nue)*Exy[k + n3 * (j + n2 * (i))])/denominator_e;
        fix[k + n3 * (j + n2 * (i))]=(0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*mui*Ex[k + n3 * (j + n2 * (i))]-difi*difxni[k + n3 * (j + n2 * (i))]
        -wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui+wci*q*0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui))/denominator_i;
    }


    for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;

        if (j * k == 0 ||  j >= jmax-1 || k >= kmax-1) continue;

        if (fex[k + n3 * (j + n2 * (0))] > 0.0){
                fex[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (0))] > 0.0){
                fix[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fex[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fex[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fix[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------

        // Now we calculate the fluxes in z direction
      for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fez[k + n3 * (j + n2 * (i))]=-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k+1 + n3 * (j + n2 * (i))])*mue*Ez[k + n3 * (j + n2 * (i))]-dife*difzne[k + n3 * (j + n2 * (i))];
        fiz[k + n3 * (j + n2 * (i))]=0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k+1 + n3 * (j + n2 * (i))])*mui*Ez[k + n3 * (j + n2 * (i))]-difi*difzni[k + n3 * (j + n2 * (i))];

    }
       // BC on fluxes

    for ( I = index_x; I < n1 * n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1) continue;
        if (fez[0 + n3 * (j + n2 * (i))]>0.0){
            fez[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[0 + n3 * (j + n2 * (i))]>0.0){
            fiz[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fez[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fez[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fiz[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
    }
// -----------------------------------------------------------------------------------------------


       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]-dt*(fex[k + n3 * (j + n2 * (i))]-fex[k + n3 * (j + n2 * (i-1))]+fey[k + n3 * (j + n2 * (i))]-fey[k + n3 * (j-1 + n2 * (i))]+fez[k + n3 * (j + n2 * (i))]-fez[k-1 + n3 * (j + n2 * (i))])/h ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]-dt*(fix[k + n3 * (j + n2 * (i))]-fix[k + n3 * (j + n2 * (i-1))]+fiy[k + n3 * (j + n2 * (i))]-fiy[k + n3 * (j-1 + n2 * (i))]+fiz[k + n3 * (j + n2 * (i))]-fiz[k-1 + n3 * (j + n2 * (i))])/h ;
    }





        for ( I = index_x; I < n1 * n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax || j >= jmax ) continue;

        ne[0 + n3 * (j + n2 * (i))] = -dt*fez[0 + n3 * (j + n2 * (i))]/h ;
        ni[0 + n3 * (j + n2 * (i))] = -dt*fiz[0 + n3 * (j + n2 * (i))]/h ;

    }

     for ( I = index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = -dt*fey[k + n3 * (0 + n2 * (i))]/h ;
        ni[k + n3 * (0 + n2 * (i))] = -dt*fiy[k + n3 * (0 + n2 * (i))]/h ;

    }




      for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= -dt*fex[k + n3 * (j + n2 * (0))]/h ;
       ni[k + n3 * (j + n2 * (0))]= -dt*fix[k + n3 * (j + n2 * (0))]/h ;


    }




        // BC on densities

     for ( I = index_x; I < n1 * n3; I += stride_x) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (0 + n2 * (i))] = 0.0 ;

        ne[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;

    }




      for ( I = index_x; I < n2 * n3; I += stride_x) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= 0.0 ;
       ni[k + n3 * (j + n2 * (0))]= 0.0 ;

       ne[k + n3 * (j + n2 * (imax-1))]= 0.0 ;
       ni[k + n3 * (j + n2 * (imax-1))]= 0.0 ;


    }



    for ( I = index_x; I < n1*n2; I += stride_x) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax+1 || j >= jmax+1) continue;
        ne[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ne[0 + n3 * (j + n2 * (i))]=0.0;
        ni[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ni[0 + n3 * (j + n2 * (i))]=0.0;

    }




        // calculating the loss
         sf=0.0;
       for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
         if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        sf=sf+ne[k + n3 * (j + n2 * (i))] ;
    }

        alpha=(si-sf)/sf;

    for ( I = index_x; I < N; I += stride_x) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
    }
      // if (myTime%100==0.0){
     //  }

}
void mardas(int imax, int jmax, int kmax, int tmax, float *ne, float *ni, float *difxne, float *difyne, float *difxni, float *difyni, float *difxyne, float *difxyni, float *Exy, float *fexy, float *fixy, float *g, float* g_temp, float *R, float *Ex, float *Ey, float *fex, float *fey, float *fix, float *fiy, float *V, float *L, float *difzne, float *difzni, float *Ez, float *fez, float *fiz, float qi, float qe, float kr, float ki, float si, float sf, float alpha, float q, float pie, float Ta , float w , float eps0 , float Te, float Ti, float B, float Kb, float me, float mi, float nue, float nui, float denominator_e, float denominator_i, float nn, float dt, float h, float wce, float wci, float mue, float mui, float dife, float difi) {
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1; 
    N=n1*n2*n3;
float** kernel = create_kernel(imax, jmax, kmax, w);
sparseMat* sp = dense_to_sparse(N, kernel);
float** A;
float** BB;
float** holder1;
float* holder2;
cudaMallocManaged(&(A ), N*sizeof(A));
cudaMallocManaged(&(BB ), N*sizeof(BB));
cudaMallocManaged(&(holder1 ), N*sizeof(holder1));
cudaMallocManaged(&(holder2 ), N*sizeof(holder2));
for (int i=0; i<N; i++) {
    //cudaMallocManaged(&my_mat[i], N*sizeof(my_mat[i]));
    cudaMallocManaged(&(A[i] ), N*sizeof(A[i]));
    cudaMallocManaged(&(BB[i] ), N*sizeof(BB[i]));
    cudaMallocManaged(&(holder1[i] ), N*sizeof(holder1[i]));
}
int iterations = 40;
//sparseMats* AB = createAB(N, kernel, iterations, A, BB, holder1);
sparseMats* AB = readAB(n1, n2, n3, iterations, A, BB);

double begin = clock();
for ( myTime=1; myTime<tmax; myTime++){  // This for loop takes care of myTime evolution
     fuckingCount = 0;
     printf("%d\n", myTime);

    before_cu<<<N, 1>>>(imax, jmax, kmax, tmax, ne, ni, difxne, difyne, difxni, difyni, difxyne, difxyni, Exy, fexy, fixy, g, g_temp, R, Ex, Ey, fex, fey, fix, fiy, V, L, difzne, difzni, Ez, fez, fiz, qi, qe, kr, ki, si, sf, alpha, q, pie,Ta ,w ,eps0 , Te, Ti, B, Kb, me, mi, nue, nui, denominator_e, denominator_i, nn, dt, h, wce, wci, mue, mui, dife, difi);

    //poisson_solve(imax, jmax, kmax, n1, n2, n3, N, iterations, V, g, R, w, h);
    //poisson_solve2(imax, jmax, kmax, n1, n2, n3, N, iterations, V, g, g_temp, R, w, h, sp, kernel);
    poisson_solve3_cu<<<N, 1>>>(imax, jmax, kmax, n1, n2, n3, N, iterations, V, g, g_temp, R, w, h, AB, holder2);


    after_cu<<<N, 1>>>(imax, jmax, kmax, tmax, ne, ni, difxne, difyne, difxni, difyni, difxyne, difxyni, Exy, fexy, fixy, g, g_temp, R, Ex, Ey, fex, fey, fix, fiy, V, L, difzne, difzni, Ez, fez, fiz, qi, qe, kr, ki, si, sf, alpha, q, pie,Ta ,w ,eps0 , Te, Ti, B, Kb, me, mi, nue, nui, denominator_e, denominator_i, nn, dt, h, wce, wci, mue, mui, dife, difi);


     }
    cudaDeviceSynchronize();
    double time_spent1 = (clock() - begin) / CLOCKS_PER_SEC;
    printf("Time spent without parallelization: %f\n", time_spent1);
}

int main()
{
int imax = 16, jmax = 16, kmax = 16,i,j,k;
int n1 = imax+3, n2 = jmax+3, n3 = kmax+3;
float qi=1.6E-19,qe=-1.6E-19, kr = 0,ki = 0,si = 0,sf = 0,alpha = 0, q=1.6E-19,pie=3.14159,Ta,w,eps0,Te,Ti,B,Kb,me,mi,nue,nui,denominator_e,denominator_i,nn,dt,h,wce,wci,mue,mui,dife,difi;
int tmax = 200;
float *ne;
float *ni;
float *difxne;
float *difyne;
float *difxni;
float *difyni;
float *difxyne;
float *difxyni;
float *Exy;
float *fexy;
float *fixy;
float *g;
float *g_temp;
float *R;
float *Ex;
float *Ey;
float *fex;
float *fey;
float *fix;
float *fiy;
float *V;
float *L;
float *difzne;
float *difzni;
float *Ez;
float *fez;
float *fiz;
    cudaMallocManaged(&(ne ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(ni ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difxne ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difyne ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difxni ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difyni ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difxyne ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difxyni ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(Exy ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fexy ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fixy ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(g ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(g_temp ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(R ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(Ex ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(Ey ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fex ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fey ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fix ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fiy ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(V ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(L ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difzne ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(difzni ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(Ez ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fez ), n1 * n2 * n3 * sizeof(float));
    cudaMallocManaged(&(fiz ), n1 * n2 * n3 * sizeof(float));

    Kb    = 1.38E-23;
    B     = 0.0;
    Te    = 2.5*11604.5;
    Ti    = 0.025*11604.5;
    me    = 9.109E-31;
    mi    = 6.633E-26;
    ki    = 0.0;
    dt    = 1.0E-12;
    h     = 1.0E-3;
    eps0  = 8.854E-12;
    si    = 0.0;
    sf    =0.0;



    for ( i=0; i<imax+3;i++){
        for ( j=0; j<jmax+3; j++){
            for ( k=0; k<kmax+3;k++){
                ne[k + n3 * (j + n2 * (i))] = 1e-9;
                ni[k + n3 * (j + n2 * (i))] = 1e-9;
                difxne[k + n3 * (j + n2 * (i))] = 1e-9;
                difyne[k + n3 * (j + n2 * (i))] = 1e-9;
                difxni[k + n3 * (j + n2 * (i))] = 1e-9;
                difyni[k + n3 * (j + n2 * (i))] = 1e-9;
                difxyne[k + n3 * (j + n2 * (i))] = 1e-9;
                difxyni[k + n3 * (j + n2 * (i))] = 1e-9;
                Exy[k + n3 * (j + n2 * (i))] = 1e-9;
                fexy[k + n3 * (j + n2 * (i))] = 1e-9;
                fixy[k + n3 * (j + n2 * (i))] = 1e-9;
                g[k + n3 * (j + n2 * (i))] = 1e-9;
                R[k + n3 * (j + n2 * (i))] = 1e-9;
                Ex[k + n3 * (j + n2 * (i))] = 1e-9;
                Ey[k + n3 * (j + n2 * (i))] = 1e-9;
                fex[k + n3 * (j + n2 * (i))] = 1e-9;
                fey[k + n3 * (j + n2 * (i))] = 1e-9;
                fix[k + n3 * (j + n2 * (i))] = 1e-9;
                fiy[k + n3 * (j + n2 * (i))] = 1e-9;
                V[k + n3 * (j + n2 * (i))] = (rand() % 10)/10.;
                L[k + n3 * (j + n2 * (i))] = 1e-9;
                difzne[k + n3 * (j + n2 * (i))] = 1e-9;
                difzni[k + n3 * (j + n2 * (i))] = 1e-9;
                Ez[k + n3 * (j + n2 * (i))] = 1e-9;
                fez[k + n3 * (j + n2 * (i))] = 1e-9;
                fiz[k + n3 * (j + n2 * (i))] = 1e-9;
             }
        }
    }


    nn=1.33/(Kb*Ti); //neutral density=p/(Kb.T)
    nue=nn*1.1E-19*sqrt(Kb*Te/me); // electron collision frequency= neutral density * sigma_e*Vth_e
    nui=nn*4.4E-19*sqrt(Kb*Ti/mi);
    wce=q*B/me;
    wci=q*B/mi;
    mue=q/(me*nue);
    mui=q/(mi*nui);
    dife=Kb*Te/(me*nue);
    difi=Kb*Ti/(mi*nui);
    ki=0.00002/(nn*dt);
    denominator_e= (1+wce*wce/(nue*nue));
    denominator_i= (1+wci*wci/(nui*nui));
    // Ta and W are just some constants needed for the iterative method that we have used to solve Poisson eq.
    Ta=acos((cos(pie/imax)+cos(pie/jmax)+cos(pie/kmax))/3.0);// needs to be float checked
    w=2.0/(1.0+sin(Ta));
// -----------------------------------------------------------------------------------------------
      //Density initialization
      // To add multiple Gaussian sources, just simply use the density_initialization function at the (x,y) points that you want
        int x_position = 15, y_position = 15, z_position = 15;
       for ( i=1; i<imax-1;i++){
        for ( j=1; j<jmax-1;j++){
            for ( k=1; k<kmax-1;k++){
                ne[k + n3 * (j + n2 * (i))]= 2.0E13;/*
                    1.0E14+1.0E14*exp(-(pow((i-x_position),2)+
                    pow((j-y_position),2)+pow((k-z_position),2))/100.0);*/
                ni[k + n3 * (j + n2 * (i))]=2.0E13;/*
                    1.0E14+1.0E14*exp(-(pow((i-x_position),2)+
                    pow((j-y_position),2)+pow((k-z_position),2))/100.0);*/
            }
        }
    }



        for ( k=1; k<kmax+1; k++) {
          for ( j=1; j<jmax+1; j++) {
           for ( i=1; i<imax+1;i++) {
         si=si+ne[k + n3 * (j + n2 * (i))] ;
          }
         }
        }
// -----------------------------------------------------------------------------------------------
mardas(imax, jmax, kmax, tmax, ne, ni, difxne, difyne, difxni, difyni, difxyne, difxyni, Exy, fexy, fixy, g, g_temp, R, Ex, Ey, fex, fey, fix, fiy, V, L, difzne, difzni, Ez, fez, fiz, qi, qe, kr, ki, si, sf, alpha, q, pie,Ta ,w ,eps0 , Te, Ti, B, Kb, me, mi, nue, nui, denominator_e, denominator_i, nn, dt, h, wce, wci, mue, mui, dife, difi);




    printf("%f\n", V[5 + (kmax+3) * (5 + (jmax+3) * 5)]);
 //   printf("%f\n", V[my_index + (kmax+3) * (my_index + (jmax+3) * my_index)]);

}
// -----------------------------------------------------------------------------------------------
