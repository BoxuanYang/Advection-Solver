// CUDA parallel 2D advection solver module
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters
#include <cuda_runtime.h>

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
  // this is a copy of the update_advection_field() function.
  // I met errors when directly use update_advection_field(), so I create a local copy
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
  // calculate the size of each halo that the thread needs to work on 
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
  // calculate the size of each halo that the thread needs to work on 
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
  // calculate the size of matrix each thread needs to work on
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

__global__
void update_advection_kernel_optimized(int M, int N, double *u, int ldu, double *v, int ldv, double Ux, double Uy){
  // TODO
  double cim1, ci0, cip1;
  double cjm1, cj0, cjp1;
  calculate_and_update_coefficients(Ux, &cim1, &ci0, &cip1);
  calculate_and_update_coefficients(Uy, &cjm1, &cj0, &cjp1);

  int Gx = gridDim.x;
  int Gy = gridDim.y;
  int Bx = blockDim.x;
  int By = blockDim.y;

  int maxM = (M  + Gx - 1) / Gx;
  int maxN = (N  + Gy - 1) / Gy;
  int sharedMemSize = maxM * (maxN);
  //printf("sharedMemSize: %d\n", sharedMemSize);

  // Compute the size of the submatrix that the block need to work on in M dimension, halo excluded
  // Also, compute the starting index of the submatrix in M dimension in the u matrix
  int block_size_M = (M + Gx - 1) / Gx;
  int block_M_start = blockIdx.x * block_size_M + 1;
  if(M % block_size_M != 0){
    block_size_M = (blockIdx.x < Gx - 1) ? block_size_M : M % block_size_M;
  }

  // Compute the size of the submatrix that the block need to work on in N dimension, halo excluded
  // Also, compute the starting index of the submatrix in N dimension in the u matrix
  int block_size_N = (N + Gy - 1) / Gy;
  int block_N_start = blockIdx.y * block_size_N + 1;
  if(N % block_size_N != 0){
    block_size_N = (blockIdx.y < Gy - 1) ? block_size_N : N % block_size_N;
  }

  int block_size = block_size_M * block_size_N;
  if(block_size > sharedMemSize){
    printf("Too much memory for block (%d,%d)\n", blockIdx.x, blockIdx.y);
  }
  

  // Compute the size of the submatrix that the thread need to work on in M dimension, halo excluded
  // Also, compute the starting index of the submatrix in M dimension in the u matrix
  int thread_size_M = block_size_M / Bx;
  // starting index of the thread in u equals block_M_start + offset
  int thread_M_start = block_M_start + threadIdx.x * thread_size_M;
  thread_size_M = (threadIdx.x < Bx - 1) ? thread_size_M : block_size_M - thread_size_M * (Bx - 1);

  // Compute the size of the submatrix that the thread need to work on in N dimension, halo excluded
  // Also, compute the starting index of the submatrix in M dimension in the u matrix
  int thread_size_N = block_size_N / By;
  // starting index of the thread in u equals block_N_start + offset
  int thread_N_start = block_N_start + threadIdx.y * thread_size_N;
  thread_size_N = (threadIdx.y < By - 1) ? thread_size_N : block_size_N - thread_size_N * (By - 1);
  


  // sharedMem is a shared memory, where shraedMem[i][j] where i, j represents u[block_M_start + i][block_N_start + j]
  extern __shared__ double sharedMem[];

  // Update sharedMem array
  // i, j here refers to the indexes as in matrix u
  for(int i = thread_M_start - 1; i < thread_M_start + thread_size_M + 1; i++){
    int s_i = i - block_M_start + 1; // i.e., thread_M_start - block_M_start at the beginning
    for(int j = thread_N_start; j < thread_N_start + thread_size_N; j++){
      int s_j = j - block_N_start;
      if(0 <= i && i <= M + 1 && 1 <= j && j <= N){
        sharedMem[s_i * block_size_N + s_j] = (cjm1 * u[i * ldu + j - 1] + cj0 * u[i * ldu + j] + cjp1 * u[i * ldu + j + 1]);
      }
    }
  }

  
  // perform advection update for a thread
  __syncthreads();
  for(int i = thread_M_start; i < thread_M_start + thread_size_M; i++){
    int s_i = i - block_M_start + 1;
    for(int j = thread_N_start; j < thread_N_start + thread_size_N; j++){
      int s_j = j - block_N_start;
      if(1 <= i && i <= M && 1 <= j && j <= N){
        v[i * ldv + j] = 
        cim1 * sharedMem[(s_i - 1) * block_size_N + s_j] +
        ci0 * sharedMem[s_i * block_size_N + s_j] +
        cip1 * sharedMem[(s_i + 1) * block_size_N + s_j];
      }
      
    }
  } 
  __syncthreads(); 

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

__global__ void emptyKernel(){
  return;
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

  /*
  
  int numKernels = 10000;
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for(int i = 0; i < numKernels; i++){
    emptyKernel<<<1, 1>>>();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("Kernel lauch time: %f ms \n", time / numKernels);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceReset(); */
  

  for(int r = 0; r < reps; r++){
    // update top/bottom halo
    update_top_bot_halo_kernel<<<dimG, dimB>>>(M, N, d_u, ldu);
    // update left/right halo
    update_left_right_halo_kernel<<<dimG, dimB>>>(M, N, d_u, ldu);
    // update advcetion
    update_advection_kernel<<<dimG, dimB>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);
    // copy back
    copy_field_kernel<<<dimG, dimB>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);
  }

  cudaMemcpy(u, d_u, ldu * (M + 2) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  cudaFree(d_v);
} //run_parallel_cuda_advection_2D_decomposition()



// ... optimized parallel variant
void run_parallel_cuda_advection_optimized(int reps, double *u, int ldu, int w) {
  double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
  int ldv = N + 2; 
  double *d_v, *d_u;;

  cudaMalloc(&d_v, ldv * (M + 2) * sizeof(double));
  cudaMalloc(&d_u, ldu * (M + 2) * sizeof(double));

  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);

  cudaMemcpy(d_u, u, ldv * (M + 2) * sizeof(double), cudaMemcpyHostToDevice);
  
  //printf("M: %d, N: %d\n", M, N);
  

  for(int r = 0; r < reps; r++){
    // update top/bottom halo
    update_top_bot_halo_kernel<<<dimG, dimB>>>(M, N, d_u, ldu);

    // update left/right halo
    update_left_right_halo_kernel<<<dimG, dimB>>>(M, N, d_u, ldu);

    // update advcetion
    int max_M = (M + Gx - 1) / Gx;
    int max_N = (N + Gy - 1) / Gy;
    int sharedMemSize = (max_M + 2) * (max_N + 2) * sizeof(double);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(sharedMemSize > prop.sharedMemPerBlock){
      printf("The max shared memory per block is: %zu bytes\n", prop.sharedMemPerBlock);
      printf("Shared memory size is: %d bytes\n", sharedMemSize);
    }

    printf("Doing optimization\n");
    update_advection_kernel_optimized<<<dimG, dimB, sharedMemSize>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // copy back
    double *tmp = d_u;
    d_u = d_v;
    d_v = tmp;
    //copy_field_kernel<<<dimG, dimB>>>(M, N, d_u, ldu, d_v, ldv, Ux, Uy);
  }

  cudaMemcpy(u, d_u, ldu * (M + 2) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  cudaFree(d_v);
  
} //run_parallel_cuda_advection_optimized()