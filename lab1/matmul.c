#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define KILO  1024
#define MEGA  KILO * KILO
#define L1_CACHE_SIZE 32 * KILO
#define L2_CACHE_SIZE 256 * KILO
#define L3_CACHE_SIZE 12 * MEGA

// You can use #define below to choose which algorithm you are use
#define SWAP 1        // Swap order
#define FULL_AWARE 0  // Consider L2 and L3 cache sizes
#define AWARE 1       // Regular Cache aware
#define OBLIV 0       // Regular Cache oblivious
                      // If none of above is 1, run original implementation
#define TEST  0       // For testing

// calculate C = AxB
// 1. Original Implementation
void matmul(float **A, float **B, float **C, int m, int n, int p) {
  int   i, j, k;

#if SWAP
  float temp;
	for (i=0; i<m; i++) {
  // for each row of C
    for (k=0; k<p; k++) {
      temp = A[i][k];
  	  for (j=0; j<n; j++) {
    // for each column of C
      // dot product of row from A and column from B
        C[i][j] += temp * B[k][j];
      }
    }
  }
#else
	float sum;
	for (i=0; i<m; i++) {
  // for each row of C
  	for (j=0; j<n; j++) {
    // for each column of C
    	sum = 0.0f; // temporary value
      for (k=0; k<p; k++) {
      // dot product of row from A and column from B
      	sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
#endif
}

void matmul_base(float **A, float **B, float **C, int m0, int n0, int p0, int dx) {
#if TEST
  printf("l0: m0 = %d, n0 = %d, p0 = %d, dx = %d\n", m0, n0, p0, dx);
#endif
  int i, j, k, m1, n1, p1;
  m1 = m0 + dx;
  n1 = n0 + dx;
  p1 = p0 + dx;
#if SWAP
  float temp;
  for (i=m0; i<m1; i++) {
    for (k=p0; k<p1; k++) {
      temp = A[i][k];
      for (j=n0; j<n1; j++) {
        C[i][j] += temp * B[k][j];
      }
    }
  }
#else
  float sum;
  for (i=m0; i<m1; i++) {
    for (j=n0; j<n1; j++) {
      sum = 0.0f; // temporary value
      for (k=p0; k<p1; k++) {
      	sum += A[i][k]*B[k][j];
      }
      C[i][j] += sum;
    }
  }
#endif
}

// 2. Cache-aware Implementation
void matmul_l1_aware(float **A, float **B, float **C, int m0, int n0, int p0, int dx) {
#if TEST
  printf("l1: m0 = %d, n0 = %d, p0 = %d, dx = %d\n", m0, n0, p0, dx);
#endif
  int i, j, k, mm, nn, pp, m1, n1, p1;
  unsigned int size = (int) sqrt(L1_CACHE_SIZE / 3);
  m1 = m0 + dx;
  n1 = n0 + dx;
  p1 = p0 + dx;
  if(size < 16) {
    size = 8; 
  }
  else if(size < 32) {
    size = 16; 
  }
  else if(size < 64) {
    size = 32; 
  }
  else if(size < 128) {
    size = 64; 
  }
  else if(size < 256) {
    size = 128; 
  }
  else if(size < 512) {
    size = 256; 
  }
  else {
    size = 512; 
  }
//  size /= 2;
  if(dx <= size) {
    matmul_base(A, B, C, m0, n0, p0, dx);
    return;
  }
  mm = dx/size;
  nn = dx/size;
  pp = dx/size;

	for (i=0; i<mm; i++) {
  // for each row of C
  	for (j=0; j<nn; j++) {
    // for each column of C
      for (k=0; k<pp; k++) {
        matmul_base(A, B, C, m0+i*size, n0+j*size, p0+k*size, size);
      }
    }
  }
}

void matmul_l2_aware(float **A, float **B, float **C, int m0, int n0, int p0, int dx) {
#if TEST
  printf("l2: m0 = %d, n0 = %d, p0 = %d, dx = %d\n", m0, n0, p0, dx);
#endif
  int i, j, k, mm, nn, pp, m1, n1, p1;
  unsigned int size = (int) sqrt(L2_CACHE_SIZE / 3);
  m1 = m0 + dx;
  n1 = n0 + dx;
  p1 = p0 + dx;
  if(size < 64) {
    size = 32; 
  }
  else if(size < 128) {
    size = 64; 
  }
  else if(size < 256) {
    size = 128; 
  }
  else if(size < 512) {
    size = 256; 
  }
  else if(size < 1024) {
    size = 512; 
  }
  else {
    size = 1024;
  }
  if(dx <= size) {
    matmul_l1_aware(A, B, C, m0, n0, p0, dx);
    return;
  }
  mm = nn = pp = dx/size;

	for (i=0; i<mm; i++) {
  // for each row of C
  	for (j=0; j<nn; j++) {
    // for each column of C
      for (k=0; k<pp; k++) {
        matmul_l1_aware(A, B, C, m0+i*size, n0+j*size, p0+k*size, size);
      }
    }
  }
}

void matmul_l3_aware(float **A, float **B, float **C, int m0, int n0, int p0, int dx) {
#if TEST
  printf("l3: m0 = %d, n0 = %d, p0 = %d, dx = %d\n", m0, n0, p0, dx);
#endif
  int i, j, k, mm, nn, pp, m1, n1, p1;
  unsigned int size = (int) sqrt(L3_CACHE_SIZE / 3);
  m1 = m0 + dx;
  n1 = n0 + dx;
  p1 = p0 + dx;
  if(size < 128) {
    size = 64; 
  }
  else if(size < 256) {
    size = 128; 
  }
  else if(size < 512) {
    size = 256; 
  }
  else if(size < 1024) {
    size = 512; 
  }
  else if(size < 2048) {
    size = 1024;
  }
  else {
    size = 2048;
  }
  if(dx <= size) {
    matmul_l2_aware(A, B, C, m0, n0, p0, dx);
    return;
  }
  mm = nn = pp = dx/size;

	for (i=0; i<mm; i++) {
  // for each row of C
  	for (j=0; j<nn; j++) {
    // for each column of C
      for (k=0; k<pp; k++) {
        matmul_l2_aware(A, B, C, m0+i*size, n0+j*size, p0+k*size, size);
      }
    }
  }
}

// 3. Cache-oblivious Implementation
void matmul_obliv(int i0, int i1,int j0, int j1, int k0, int k1, float **A, float **B, float **C) {
  int i, j, k, di = i1 - i0, dj = j1 - j0, dk = k1 - k0;
  const int CUTOFF = 32;
  if (di >= dj && di >= dk && di > CUTOFF) {
    int im = (i0 + i1) / 2;
    matmul_obliv(i0,im,j0,j1,k0,k1, A, B, C);
    matmul_obliv(im,i1,j0,j1,k0,k1, A, B, C);
  } else if (dj >= dk && dj > CUTOFF) {
    int jm = (j0 + j1) / 2;
    matmul_obliv(i0,i1,j0,jm,k0,k1, A, B, C);
    matmul_obliv(i0,i1,jm,j1,k0,k1, A, B, C);
  } else if (dk > CUTOFF) {
    int km = (k0 + k1) / 2;
    matmul_obliv(i0,i1,j0,j1,k0,km, A, B, C);
    matmul_obliv(i0,i1,j0,j1,km,k1, A, B, C);
  } else {
#if SWAP
    float temp;
    for (i = i0; i < i1; ++i) {
      for (k = k0; k < k1; ++k) {
        temp = A[i][k];
        for (j = j0; j < j1; ++j) {
          C[i][j] += temp * B[k][j];
        }
      }
    }
#else
    for (i = i0; i < i1; ++i)
      for (j = j0; j < j1; ++j)
        for (k = k0; k < k1; ++k)
          C[i][j] += A[i][k] * B[k][j];
#endif
  }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
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

void create_matrix_test(float*** A, int m, int n) {
  float **T = 0;
  int i, j;

  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
    T[i] = (float*)malloc(n*sizeof(float));
  }

  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      T[i][j] = i+j;
    }
  }
  *A = T;
}

// test functionality of matrix multiplication
void print_matrix(float **C, float **C1, int m, int n) {
  int i, j;
  FILE *file;
  file = fopen("orig_result.out", "w+");
  fprintf(file, "Result after matrix multiplication:\n");
	for (i=0; i<m; i++) {
  	for (j=0; j<n; j++) {
      fprintf(file, "%f ", C1[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);

  file = fopen("aware_result.out", "w+");
  fprintf(file, "Result after matrix multiplication:\n");
	for (i=0; i<m; i++) {
  	for (j=0; j<n; j++) {
      fprintf(file, "%f ", C[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

void free_matrix(float **T, int m) {
  int i;

  for ( i=0; i<m; i++ ) {
    free(T[i]);
  }
  free(T);
}

int main(int argc, char *argv[]) {
  float** A;
  float** B;
  float** C;
  int M, N, P;
  M = N = P = 0;
  
  if (argc < 2 || argc > 4) {
    printf("ERROR: Not enough of arguments\n");
    printf("USAGE: ./matmul M [N P]\n");
  }
  else if (argc == 2) {
    M = N = P = atoi(argv[1]);
  }
  else if (argc == 3) {
    M = P = atoi(argv[1]);
    N = atoi(argv[2]);
  }
  else if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    P = atoi(argv[3]);
  }

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);
#if TEST
  float **C1;
  create_matrix(&C1, M, N);
  int i, j;
	for (i=0; i<M; i++) {
  	for (j=0; j<P; j++) {
      A[i][j] = j%10;
    }
  }
	for (i=0; i<P; i++) {
  	for (j=0; j<N; j++) {
      B[i][j] = j%10;
    }
  }
	for (i=0; i<M; i++) {
  	for (j=0; j<N; j++) {
      C[i][j] = 0;
      C1[i][j] = 0;
    }
  }
#endif

	// assume some initialization of A and B
	// think of this as a library where A and B are 
	// inputs in row-major format, and C is an output 
	// in row-major.  
#if FULL_AWARE
  matmul_l3_aware(A, B, C, 0, 0, 0, M);
#elif AWARE
  matmul_l1_aware(A, B, C, 0, 0, 0, M);
#elif OBLIV
  matmul_obliv(0, M-1, 0, N-1, 0, P-1, A, B, C);
#else
  matmul(A, B, C, M, N, P);
#endif
#if TEST
  matmul(A, B, C1, M, N, P);
  print_matrix(C, C1, M, N);
  free_matrix(C1, M);
#endif
  free_matrix(A, M);
  free_matrix(B, P);
  free_matrix(C, M);

  return 0;
}

