Computer Architecture: Parallelism & Locality Lab Description
=============================================================
Instructor: Prof. Mattan Erez

From smart phones, to multi-core CPUs and GPUs, to the world's largest supercomputers and web sites, parallel processing is ubiquitous in modern computing. The course provide a deep understanding of the fundamental principles and engineering trade-offs involved in designing modern parallel computing systems and teach parallel programming techniques necessary to effectively utilize these machines. Because writing good parallel programs requires an understanding of key machine performance characteristics, this course will cover both parallel hardware and software design.

Lab1. Locality in a CPU
-----------------------
In this lab, we use dense matrix-matrix multiplication to study the effects of locality on performance and power dissipation. We explore two programming styles in the domain of dense linear algebra: iterative and recursive with cache-aware and cache-oblivious algorithm and learn about 3 tools that can help with architecture optimizations (performance counters, CACTI and PIN).
    
    // Baseline matrix-matrix multiplication
    #include <stdlib.h>
    
    // define the matrix dimensions A is MxP, B is PxN, and C is MxN
    #define M 512
    #define N 512
    #define P 512
    
    // calculate C = AxB
    void matmul(float **A, float **B, float **C) {
      float sum;
      int   i;
      int   j;
      int   k;
    
      for (i=0; i<M; i++) {
        // for each row of C
        for (j=0; j<N; j++) {
          // for each column of C
          sum = 0.0f; // temporary value
          for (k=0; k<P; k++) {
            // dot product of row from A and column from B
            sum += A[i][k]*B[k][j];
          }
          C[i][j] = sum;
        }
      }
    }
    
    // function to allocate a matrix on the heap
    // creates an mXn matrix and returns the pointer.
    //
    // the matrices are in row-major order.
    void create_matrix(float*** A, int m, int n) {
      float **T = 0;
      int i;
    
      T = (float**)malloc( m*sizeof(float*));
      for ( i=0; i<m; i++ ) {
         T[i] = (float*)malloc(n*sizeof(float));
      }
      *A = T;
    }
    
    int main() {
      float** A;
      float** B;
      float** C;
    
      create_matrix(&A, M, P);
      create_matrix(&B, P, N);
      create_matrix(&C, M, N);
    
      // assume some initialization of A and B
      // think of this as a library where A and B are
      // inputs in row-major format, and C is an output
      // in row-major.
    
      matmul(A, B, C);
    
      return (0);
    }

Step1: Improve the performance of the baseline version above on a uni-processor. Most of the performance to be gained is with locality optimizations, followed by converting the code to use modern processor's short-vector SIMD units. 

Step2: Use the performance counters to measure the number of cycles, instructions and cache hits and misses required by our application to calculate GFLOPS. (Although PAPI package is the best supported, we use perf tool instead for the convenience of teaching.) 

Step3: Try both cache-aware and cache-oblivious methods to improve locality and hence performance (optimizing registers as well as the memory hierarchy).

Step4: Modify existing PIN tools to implement a read- and write-allocate, write-back, inclusive 2-level cache model with LRU replacement.

Step5: Use CACTI tool to estimate the power and energy consumed by the matrix multiplication and the potential power savings of locality optimizations.

[Here](./lab1) are codes of Lab1.


