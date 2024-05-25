// OpenMP parallel 2D advection solver module

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q;
static int verbosity;

//sets up parameters above
void init_parallel_parameter_values(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; 
  P = P_, Q = Q_;
  verbosity = verb;
} //init_parallel_parameter_values()

/*
This function should be parallelized with the purpose of improving performance.
*/
void omp_update_boundary_1D_decomposition(double *u, int ldu) {
  /* Original code
  int i,j;
  for (j = 1; j < N+1; j++) { //top and bottom halo
    u[j] = u[M * ldu + j];
    u[(M + 1) * ldu + j] = u[ldu + j];
  }

  for (i = 0; i < M+2; i++) { //left and right sides of halo 
    u[i * ldu] = u[i * ldu + N];
    u[i * ldu + N + 1] = u[i * ldu + 1];
  }
  */
  
  

  int upper = (M > N) ? M : N;
  int i;

  #pragma omp parallel for schedule(static)
  for(i = 1; i <= upper; i++){
    if(i <= N){
      u[i] = u[M * ldu + i];
      u[(M + 1) * ldu + i] = u[ldu + i];
    }

    if(i <= M){
      u[i * ldu] = u[i * ldu + N];
      u[i * ldu + N + 1] = u[i * ldu + 1];
    }
  }

  u[0] = u[N];
  u[(M+1) * ldu] = u[(M+1) * ldu + N];
  u[N+1] = u[1];
  u[(M+1) * ldu + N + 1] = u[(M+1) * ldu + 1];

  return;



} 

void omp_update_advection_field_1D_decomposition(double *u, int ldu, double *v, int ldv) {
  int i, j;
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  double cim1, ci0, cip1, cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1); 
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  /*
  We need to iterate all the combinations of:
  1. Parallize out/inner loop
  2. Interchange loop order
  3. Schedule iterations with block/cyclic fashion

  The openmp directive for combination 3 is:
  #pragma omp parallel for schedule(static)
  #pragma omp parallel for schedule(dynamic)

  M,N are global variables
  */
  

   // Unparallized version
  for (i=0; i < M; i++){
    for (j=0; j < N; j++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  

  /*
  1. outer loop, i-j loop, dynamic
  #pragma omp parallel for private(j) schedule(dynamic)
  for (i=0; i < M; i++){
    for (j=0; j < N; j++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */
  
  /*
  2. outer loop, i-j loop, static
  #pragma omp parallel for private(j) schedule(static)
  for (i=0; i < M; i++){
    for (j=0; j < N; j++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  3. outer loop, j-i loop, dynamic
  #pragma omp parallel for private(i) schedule(dynamic)
  for (j=0; j < N; j++){
    for (i=0; i < M; i++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  4. outer loop, j-i loop, static
  #pragma omp parallel for private(i) schedule(static)
  for (j=0; j < N; j++){
    for (i=0; i < M; i++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  5. inner loop, i-j loop, dynamic
  for (i=0; i < M; i++){
    #pragma omp parallel for schedule(dynamic)
    for (j=0; j < N; j++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  6. inner loop, i-j loop, static
  for (i=0; i < M; i++){
    #pragma omp parallel for schedule(static)
    for (j=0; j < N; j++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  7. inner loop, j-i loop, dynamic
  for (j=0; j < N; j++){
    #pragma omp parallel for schedule(dynamic)
    for (i=0; i < M; i++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

  /*
  8. inner loop, j-i loop, static
  for (j=0; j < N; j++){
    #pragma omp parallel for schedule(static)
    for (i=0; i < M; i++){
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);
    }
  }
  */

} //omp_update_advection_field_1D_decomposition()  


void omp_copy_field_1D_decomposition(double *in, int ldin, double *out, int ldout) {
  int i, j;
  #pragma omp for private(j) schedule(static)
  for (i=0; i < M; i++){
    for (j=0; j < N; j++){
      out[i * ldout + j] = in[i * ldin + j];
    }
  }
} //omp_copy_field_1D_decomposition()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void run_parallel_omp_advection_1D_decomposition(int reps, double *u, int ldu) {
  int r, ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  for (r = 0; r < reps; r++) {    
    omp_update_boundary_1D_decomposition(u, ldu);
    omp_update_advection_field_1D_decomposition(&u[ldu + 1], ldu, &v[ldv + 1], ldv);
    omp_copy_field_1D_decomposition(&v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } //for (r...)
  free(v);
} //run_parallel_omp_advection_1D_decomposition()


// ... using 2D parallelization
void run_parallel_omp_advection_2D_decomposition(int reps, double *u, int ldu) {
  int ldv = N+2;
  double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
  
  /* // Original code
  for (r = 0; r < reps; r++) {         
    for (j = 1; j < N+1; j++) { //top and bottom halo
      u[j] = u[M * ldu + j];
      u[(M + 1) * ldu + j] = u[ldu + j];
    }
    for (i = 0; i < M+2; i++) { //left and right sides of halo 
      u[i * ldu] = u[i * ldu + N];
      u[i * ldu + N + 1] = u[i * ldu + 1];
    }

    // v := u
    update_advection_field(M, N, &u[ldu + 1], ldu, &v[ldv + 1], ldv);

    // u := v
    copy_field(M, N, &v[ldv + 1], ldv, &u[ldu + 1], ldu);
  } //for (r...)
  free(v);
  */



  #pragma omp parallel 
  {
    int thread_id = omp_get_thread_num();

    // number of rows for each thread except last one
    int M_size = M / P;

    // number of cols for each thread except last one 
    int N_size = N / Q;

    // Compute the 2D thread indices
    int P0 = thread_id / Q;
    int Q0 = thread_id % Q;

    // Compute the starting row & col index of each thread, as well as 
    // their row length and col length.
    int M_start = P0 * M_size + 1;
    int M_len = P0 < P - 1 ? M_size : M - M_start + 1;
    //int M_end = M_start + M_len - 1;

    int N_start = Q0 * N_size + 1;
    int N_len = Q0 < Q - 1 ? N_size : N - N_start + 1;
    //int N_end = N_start + N_len - 1;


    //printf("Thread id: %d. (P0, Q0): (%d, %d)\n", thread_id, P0, Q0);
    //printf("M_len, N_len: (%d,%d)\n", M_len, N_len);
    //printf("M_start, N_start: (%d,%d). M_end, N_end: (%d, %d)\n\n", M_start, N_start, M_end, N_end);
    

    for(int r = 0; r < reps; r++){
      #pragma omp single
      {
        // update top bottom halo, index: [M_start][N_start] - [M_start][N_end]
        #pragma omp task
        {
          for (int j = 1; j < N+1; j++) {
            u[j] = u[M * ldu + j];
            u[(M + 1) * ldu + j] = u[ldu + j];
            }
        }

        // update left right halo
        #pragma omp task
        {
          for (int i = 0; i < M+2; i++) {
            u[i * ldu] = u[i * ldu + N];
            u[i * ldu + N + 1] = u[i * ldu + 1];
            }
        }
        

      }
      
      
      // update advection
      #pragma omp barrier
      update_advection_field(M_len, N_len, &u[M_start * ldu + N_start], ldu, &v[M_start * ldu + N_start], ldv);

      // copy back
      #pragma omp barrier
      copy_field(M_len, N_len, &v[M_start * ldu + N_start], ldv, &u[M_start * ldu + N_start], ldu);
      #pragma omp barrier
    }

  }

  free(v);
  






} //run_parallel_omp_advection_2D_decomposition()


// ... extra optimization variant
void run_parallel_omp_advection_with_extra_opts(int reps, double *u, int ldu) {

} //run_parallel_omp_advection_with_extra_opts()
