#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


double vec_dot_vec(double *v1, double *v2, int n) {
    double res=0.0;
    for (int i=0; i < n; i+=1) {
        res += v1[i] * v2[i];
    }
    return res;
}

void mat_dot_vec(int N, double I_kernel[][N], double *V, double *g_temp, double *R) {
    for (int i=0; i < N; i++) {
        R[i] = 0;
        for (int j=0; j < N; j++) {
            R[i] += I_kernel[i][j]*V[j];
        }
        R[i] -= g_temp[i];
    }
}

int main() {
    printf("mardas");
}
