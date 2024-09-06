// CUDA parallel 2D advection solver module
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters
#include <cuda_runtime.h>

__global__ void copy_field_kernel(int N, float val, float *data, int *indices){
  // TODO
  // 假设：thread block是二维的
  // N为数组长度
  int x_thread_num = gridDim.x * blockDim.x;
  int y_thread_num = gridDim.y * blockDim.y;
  int thread_num = x_thread_num * y_thread_num;

  int thread_x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_index = blockIdx.y * blockDim.y + threadIdx.y;

  int thread_index = thread_x_index * y_thread_num + thread_y_index;

  int thread_size = (N + thread_num - 1) / thread_num;
  int thread_start_index = thread_index * thread_size;
  int thread_end_index = min(thread_start_index + thread_size - 1, N - 1);

  int index = thread_index;
  // indices的长度也是N
  for(int i = thread_start_index; i <= thread_end_index; i++){
    if(data[i] > val){
        indices[index] = i;
        index++;
    }
  }
}

__global__ void emptyKernel(){
  return;
}