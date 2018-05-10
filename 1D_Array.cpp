#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/gpu/gpu.hpp>
#define my_index 3



double max1(double a, double b) {
    if (a > b) return a;
    return b;
}
void diff(int n, double *v1, double *v2) {
    double total_error = 0;
    double max_error = -1;
    int max_error_index = -1;
    for (int i = 0; i < n; i++) {
        double diff = abs(v1[i] - v2[i]);
        double maximum_value = max1(abs(v1[i]), abs(v2[i]));
        if (maximum_value < 1e-5) maximum_value = 1; 
        double rel_error = diff / maximum_value;
        if (max_error < rel_error) {
            max_error = rel_error;
            max_error_index = i;
        }
        total_error += rel_error;
    }
    printf("%12.6f\t %12.6f\t%12.6f\t%16d, %12f, %12f\n", 
            total_error, total_error/n, max_error, max_error_index,
            v1[max_error_index], v2[max_error_index]);
}







using namespace cv;

int main( int argc, char** argv )
{
      int imax = 32;
      int jmax = 32;
      int kmax = 67;
      int n1 = imax+3;
      int n2 = jmax+3;
      int n3 = kmax+3;
      int N = n1*n2*n3;
      printf("N = %d\n", N);

      float w = 1.68;

      Mat I_kernel = Mat::zeros(N, N, CV_32F);
      Mat b = I_kernel.row(1)*6;
      b.copyTo(I_kernel.row(1));
      printf("%f\n", I_kernel.at<float>(1, 2));
      double begin = clock();
      for (int I=0; I<N; I++) {
          int k = I % n3;
          int s1 = (I - k) / n3;
          int j = s1 % n2;
          int i = (s1 - j) / n2;
          if (i >= 1 && j >= 1 && k >= 1 && i < imax+1 &&
              j < jmax+1 && k < kmax+1) {
              if (I % 1000 == 0) printf("Iteration %d/%d\n", I, N);
                  Mat tmp = w / 6. *
                  (I_kernel.row(I-n2*n3) +
                    I_kernel.row(I-n2) 
                    + I_kernel.row(I-1));
                  tmp.copyTo(I_kernel.row(I));
              }
          else
              for (int j=0; j<0; j++)
                  I_kernel.at<float>(I, j) = 0;

          I_kernel.at<float>(I, I+1) += w / 6.;
          I_kernel.at<float>(I, I+n3) += w / 6.;
          I_kernel.at<float>(I, I+n2*n3) += w / 6.;
          I_kernel.at<float>(I, I) += 1 - w;
      }

      int threshold_value = 0;
      int threshold_type = 0;;
      int const max_BINARY_value = 1;
      Mat dst;
      threshold( I_kernel, dst, 0, 1, THRESH_BINARY);
      printf("sum: %f\n", sum(dst).val[0]);

      Mat A = I_kernel;
      for (int i=0; i < 40; i++) {
          printf("Iteration %d/40\n", i);
          A = I_kernel*A;
      }

      double time_spent1 = (clock() - begin) / CLOCKS_PER_SEC;
      printf("Time spent without parallelization: %f\n", time_spent1);

}
