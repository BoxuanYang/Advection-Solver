// CUDA parallel 2D advection solver module

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void init_parallel_parameter_values(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //init_parallel_parameter_values()


__host__ __device__
static void calculate_and_update_coefficients(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

__host__ __device__
void my_update_advection_field(int M, int N, double *u, int ldu, double *v, int ldv,
		       double Ux, double Uy) {
  double cim1, ci0, cip1;
  double cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      v[i * ldv + j] =
          cim1 * (cjm1 * u[(i - 1) * ldu + j - 1] + cj0 * u[(i - 1) * ldu + j] +
                  cjp1 * u[(i - 1) * ldu + j + 1]) +
          ci0 * (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] +
                 cjp1 * u[i * ldu + j + 1]) +
          cip1 * (cjm1 * u[(i + 1) * ldu + j - 1] + cj0 * u[(i + 1) * ldu + j] +
                  cjp1 * u[(i + 1) * ldu + j + 1]);

} //my_update_advection_field() 

__host__ __device__
void my_copy_field(int M, int N, double *v, int ldv, double *u, int ldu) {
  for (int i=0; i < M; i++)
    for (int j=0; j < N; j++)
      u[i * ldu + j] = v[i * ldv + j];
} // my_copy_field()

__global__ 
void update_top_bot_halo_kernel(int M, int N, double *u, int ldu){
  int x_thread_num = gridDim.x * blockDim.x;
  int y_thread_num = gridDim.y * blockDim.y;

  int thread_x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_index = blockIdx.y * blockDim.y + threadIdx.y;

  int thread_num = x_thread_num * y_thread_num;
  int thread_idx = thread_x_index * y_thread_num + thread_y_index;

  int thread_size = N / thread_num;

  int j_start = thread_idx * thread_size + 1;
  int j_end = (thread_idx < thread_num - 1) ? j_start + thread_size - 1 : N;

  for(int j = j_start; j <= j_end; j++){
    u[j] = u[M * ldu + j];
    u[(M+1) * ldu + j] = u[ldu + j];
  }

  
}

__global__ 
void update_left_right_halo_kernel(int M, int N, double *u, int ldu){
  int x_thread_num = gridDim.x * blockDim.x;
  int y_thread_num = gridDim.y * blockDim.y;

  int thread_x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_index = blockIdx.y * blockDim.y + threadIdx.y;

  int thread_num = x_thread_num * y_thread_num;
  int thread_idx = thread_x_index * y_thread_num + thread_y_index;

  int thread_size = (M+2) / thread_num;

  int i_start = thread_idx * thread_size;
  int i_end = (thread_idx < thread_num - 1) ? i_start + thread_size - 1 : M + 1;



  for(int i = i_start; i <= i_end; i++){
    u[i * ldu] = u[i * ldu + N];
    u[i * ldu + N + 1] = u[i * ldu + 1];
  }
}


// (M, N, d_u, ldu, d_v, ldv, Ux, Uy);
__global__ 
void update_advection_kernel(int M, int N, double *u, int ldu, double *v, int ldv, double Ux, double Uy){
  // TODO
  int x_thread_num = gridDim.x * blockDim.x;
  int y_thread_num = gridDim.y * blockDim.y;

  int thread_x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_index = blockIdx.y * blockDim.y + threadIdx.y;

  int M_size = M / x_thread_num;
  int M_start = thread_x_index * M_size + 1;
  int M_len = thread_x_index < x_thread_num - 1 ? M_size : M - M_start + 1;


  int N_size = N / y_thread_num;
  int N_start = thread_y_index * N_size + 1;
  int N_len = thread_y_index < y_thread_num - 1 ? N_size : N - N_start + 1;


 
  my_update_advection_field(M_len, N_len, &u[M_start * ldu + N_start], ldu, &v[M_start * ldu + N_start], ldv, Ux, Uy);





}

__global__ void copy_field_kernel(int M, int N, double *u, int ldu, double *v, int ldv, double Ux, double Uy){
  // TODO
  int x_thread_num = gridDim.x * blockDim.x;
  int y_thread_num = gridDim.y * blockDim.y;

  int thread_x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_index = blockIdx.y * blockDim.y + threadIdx.y;

  int M_size = M / x_thread_num;
  int M_start = thread_x_index * M_size + 1;
  int M_len = thread_x_index < x_thread_num - 1 ? M_size : M - M_start + 1;


  int N_size = N / y_thread_num;
  int N_start = thread_y_index * N_size + 1;
  int N_len = thread_y_index < y_thread_num - 1 ? N_size : N - N_start + 1;

  // my_copy_field(int M, int N, double *v, int ldv, double *u, int ldu)
 
  my_copy_field(M_len, N_len, &v[M_start * ldv + N_start], ldv, &u[M_start * ldv + N_start], ldu);
}




// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void run_parallel_cuda_advection_2D_decomposition(int reps, double *u, int ldu) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2; 
  double *d_v, *d_u;;

  cudaMalloc(&d_v, ldv * (M + 2) * sizeof(double));
  cudaMalloc(&d_u, ldu * (M + 2) * sizeof(double));

  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);

  cudaMemcpy(d_u, u, ldv * (M + 2) * sizeof(double), cudaMemcpyHostToDevice);

  dim3 haloG(1, 1);
  dim3 haloB(1, 1);
  
  //printf("M: %d, N: %d\n", M, N);
  

  for(int r = 0; r < reps; r++){
    // update top/bottom halo
    update_top_bot_halo_kernel<<<haloG, haloB>>>(M, N, d_u, ldu);

    // update left/right halo
    update_left_right_halo_kernel<<<haloG, haloB>>>(M, N, d_u, ldu);

    // update advcetion
    update_advection_kernel<<<haloG, haloB>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);

    // copy back
    copy_field_kernel<<<haloG, haloB>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);
  }

  cudaMemcpy(u, d_u, ldu * (M + 2) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  cudaFree(d_v);
} //run_parallel_cuda_advection_2D_decomposition()



// ... optimized parallel variant
void run_parallel_cuda_advection_optimized(int reps, double *u, int ldu, int w) {

} //run_parallel_cuda_advection_optimized()
